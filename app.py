"""
app.py
======
FastAPI service for the Machine Learning Based Crop Yield Prediction Portal.

It does two jobs:
  1. Serves the static frontend (index.html, styles.css, app.js).
  2. Serves the JSON inference API under /predict, /models and /health.

Run with:
    python app.py                 # http://127.0.0.1:8000
    uvicorn app:app --reload      # development with auto-reload

If no trained model is present the service still works using the documented
agronomic fallback, and reports that honestly via ``model_used``.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    heuristic_yield,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and artifact loading
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_xgboost.pkl"
SCHEMA_PATH = MODELS_DIR / "feature_columns.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

FALLBACK_NAME = "agronomic_fallback"
FALLBACK_R2 = 0.90


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_artifacts():
    """Load the fitted model, feature schema and metrics if they exist."""
    model = None
    feature_columns = list(FEATURE_COLUMNS)
    metrics: dict[str, Any] = {}

    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open("rb") as handle:
                model = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001 - surfaced through /health
            print(f"[warn] could not load model from {MODEL_PATH}: {exc}")
            model = None
    else:
        print(f"[warn] no trained model at {MODEL_PATH}; using {FALLBACK_NAME}")

    saved_schema = _read_json(SCHEMA_PATH)
    if isinstance(saved_schema, list) and saved_schema:
        feature_columns = [str(col) for col in saved_schema]

    saved_metrics = _read_json(METRICS_PATH)
    if isinstance(saved_metrics, dict):
        metrics = saved_metrics

    best_name, best_r2 = None, None
    for entry in metrics.get("models", []) if isinstance(metrics.get("models"), list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("r2"), (int, float)):
            if best_r2 is None or entry["r2"] > best_r2:
                best_name, best_r2 = entry.get("model"), float(entry["r2"])

    return model, feature_columns, metrics, best_name, best_r2


_model, _feature_columns, _metrics, _best_model_name, _best_r2 = load_artifacts()

app = FastAPI(
    title="Crop Yield Prediction API",
    description=(
        "Machine Learning Based Crop Yield Prediction Portal - inference service "
        "with a served static frontend."
    ),
    version="1.1.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    rainfall: float = Field(950, ge=0, le=20000, description="Rainfall in mm")
    temperature: float = Field(24, ge=-10, le=60, description="Average temperature in Celsius")
    soil_npk: float = Field(72, ge=0, le=100, description="Soil N-P-K index (0-100)")
    irrigation_hours: float = Field(30, ge=0, le=1000, description="Irrigation hours")
    farm_size_hectares: float = Field(50, gt=0, le=100000, description="Farm size in hectares")
    crop: Literal["rice", "wheat", "maize", "soybean"] = "wheat"
    season: Literal["rabi", "kharif", "zaid"] = "kharif"


class PredictionResponse(BaseModel):
    crop: str
    season: str
    predicted_yield_tph: float
    estimated_total_tonnes: float
    model_used: str
    confidence: float
    inputs: dict
    timestamp: str


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _predict_yield(payload: PredictionRequest) -> tuple[float, str, float]:
    """Return (yield_tph, model_used, confidence)."""
    inputs = payload.model_dump()

    if _model is not None:
        frame = build_feature_frame([inputs])
        frame = frame.reindex(columns=_feature_columns, fill_value=0.0)

        raw = _model.predict(frame)
        predicted = float(raw[0]) if hasattr(raw, "__len__") else float(raw)

        confidence = 0.5 + (_best_r2 if isinstance(_best_r2, float) else 0.95) * 0.5
        return max(0.0, min(50.0, predicted)), (_best_model_name or "trained_model"), round(confidence, 4)

    predicted = heuristic_yield(
        inputs["rainfall"],
        inputs["temperature"],
        inputs["soil_npk"],
        inputs["irrigation_hours"],
        inputs["crop"],
        inputs["season"],
    )
    return max(0.0, min(50.0, predicted)), FALLBACK_NAME, round(0.5 + FALLBACK_R2 * 0.5, 4)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api", tags=["System"])
def api_info():
    """Service metadata."""
    return {
        "status": "ok",
        "service": "Crop Yield Prediction API",
        "version": app.version,
        "model_loaded": _model is not None,
    }


@app.get("/health", tags=["System"])
def health_check():
    """Readiness probe reporting which engine will serve predictions."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_used": _best_model_name or FALLBACK_NAME,
        "best_r2": _best_r2,
        "feature_count": len(_feature_columns),
    }


@app.get("/models", tags=["System"])
def list_models():
    """List the model artifacts discovered on disk."""
    return {
        "active": _best_model_name or FALLBACK_NAME,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
        "schema_path": str(SCHEMA_PATH) if SCHEMA_PATH.exists() else None,
        "metrics": _metrics.get("models", []),
        "trained_at": _metrics.get("trained_at"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_yield(payload: PredictionRequest):
    """Predict crop yield (tonnes/hectare) from seasonal conditions."""
    try:
        yield_tph, model_used, confidence = _predict_yield(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert to a clean HTTP error
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    return PredictionResponse(
        crop=payload.crop,
        season=payload.season,
        predicted_yield_tph=round(yield_tph, 2),
        estimated_total_tonnes=round(yield_tph * payload.farm_size_hectares, 2),
        model_used=model_used,
        confidence=confidence,
        inputs=payload.model_dump(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Static frontend (explicit routes: only the three public files are exposed)
# ---------------------------------------------------------------------------
def _static_file(name: str) -> FileResponse:
    path = ROOT_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{name} not found")
    return FileResponse(path)


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return _static_file("index.html")


@app.get("/styles.css", include_in_schema=False)
def serve_styles() -> FileResponse:
    return _static_file("styles.css")


@app.get("/app.js", include_in_schema=False)
def serve_script() -> FileResponse:
    return _static_file("app.js")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # PORT/HOST are honoured for PAAS deployments. An empty, non-numeric or zero
    # PORT means "not configured" (uvicorn would otherwise bind a random port),
    # so we fall back to the local default. The banner is flushed because uvicorn
    # only logs the effective port after the socket is bound.
    raw_port = os.environ.get("PORT", "").strip()
    port = int(raw_port) if raw_port.isdigit() and int(raw_port) > 0 else 8000
    host = os.environ.get("HOST", "").strip() or "127.0.0.1"
    print(f"[start] Crop Yield Prediction Portal -> http://{host}:{port}", flush=True)
    print(f"[start] engine: {_best_model_name or FALLBACK_NAME}", flush=True)
    uvicorn.run(app, host=host, port=port)
