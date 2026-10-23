"""
Tests that exercise the *trained* model artifacts.

These are skipped automatically when ``models/best_xgboost.pkl`` has not been
produced yet, so a fresh clone can run the suite before training.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import MODEL_PATH, SCHEMA_PATH, app
from features import FEATURE_COLUMNS

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="trained model not found; run train_model.py first"
)

client = TestClient(app)

VALID_PAYLOAD = {
    "rainfall": 950,
    "temperature": 24,
    "soil_npk": 72,
    "irrigation_hours": 30,
    "farm_size_hectares": 50,
    "crop": "wheat",
    "season": "kharif",
}


def test_saved_schema_matches_code_schema():
    if not SCHEMA_PATH.exists():
        pytest.skip("feature schema not found")

    saved = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert saved == FEATURE_COLUMNS


def test_trained_model_serves_predictions():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["model_used"] != "agronomic_fallback"
    assert body["predicted_yield_tph"] > 0


def test_trained_model_agrees_with_reference_model():
    """A well-fit model should stay close to the agronomic reference."""
    from features import heuristic_yield

    expected = heuristic_yield(950, 24, 72, 30, "wheat", "kharif")
    predicted = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_yield_tph"]

    assert predicted == pytest.approx(expected, abs=0.25)


def test_artifacts_exist():
    assert Path(MODEL_PATH).is_file()
    assert MODEL_PATH.stat().st_size > 0
