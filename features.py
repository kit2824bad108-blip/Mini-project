"""
features.py
===========
Feature engineering shared by the training pipeline and the inference API.

Keeping this logic in one place guarantees that training and serving build
identical feature vectors — the most common source of silent ML bugs.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
CROPS = ["rice", "wheat", "maize", "soybean"]
SEASONS = ["rabi", "kharif", "zaid"]

NUMERIC_FEATURES = [
    "rainfall",
    "temperature",
    "soil_npk",
    "irrigation_hours",
    "farm_size_hectares",
]

ONE_HOT_FEATURES = [f"crop_{c}" for c in CROPS] + [f"season_{s}" for s in SEASONS]

#: Raw inputs accepted by :func:`build_feature_frame` (before one-hot encoding).
INPUT_COLUMNS = NUMERIC_FEATURES + ["crop", "season"]

#: Column order used by every model. Changing this invalidates saved models.
FEATURE_COLUMNS = NUMERIC_FEATURES + ONE_HOT_FEATURES

#: Agronomic coefficients used to synthesise the training target and as the
#: offline fallback when no trained model is available.
CROP_WEIGHTS = {"rice": 1.12, "wheat": 1.00, "maize": 0.95, "soybean": 0.88}
SEASON_WEIGHTS = {"rabi": 0.92, "kharif": 1.08, "zaid": 0.98}
BASE_YIELD_TPH = 0.80

DEFAULT_FARM_SIZE_HA = 50.0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def normalize_crop(value: str) -> str:
    """Return the canonical crop key, raising ValueError for unknown crops."""
    key = str(value or "").strip().lower()
    if key not in CROP_WEIGHTS:
        raise ValueError(f"Unknown crop {value!r}. Expected one of {', '.join(CROPS)}.")
    return key


def normalize_season(value: str) -> str:
    """Return the canonical season key, raising ValueError for unknown seasons."""
    key = str(value or "").strip().lower()
    if key not in SEASON_WEIGHTS:
        raise ValueError(f"Unknown season {value!r}. Expected one of {', '.join(SEASONS)}.")
    return key


# ---------------------------------------------------------------------------
# Yield model
# ---------------------------------------------------------------------------
def heuristic_yield(
    rainfall: float,
    temperature: float,
    soil_npk: float,
    irrigation_hours: float,
    crop: str,
    season: str,
) -> float:
    """Estimate yield (tonnes/hectare) from agronomic response factors.

    This is the reference implementation: it generates the synthetic training
    target and serves as the offline fallback predictor in the API.
    """
    crop_key = normalize_crop(crop)
    season_key = normalize_season(season)

    temp_factor = 1.0 + (float(temperature) - 20.0) * 0.012
    rain_factor = 1.0 + min(max(float(rainfall), 0.0) / 2000.0, 1.0) * 0.35
    soil_factor = 1.0 + (min(max(float(soil_npk), 0.0), 100.0) / 100.0) * 0.40
    irrigation_factor = 1.0 + min(max(float(irrigation_hours), 0.0) / 100.0, 1.0) * 0.10

    yield_tph = (
        BASE_YIELD_TPH
        * CROP_WEIGHTS[crop_key]
        * SEASON_WEIGHTS[season_key]
        * temp_factor
        * rain_factor
        * soil_factor
        * irrigation_factor
    )
    return max(yield_tph, 0.0)


def build_feature_row(
    rainfall: float = 950,
    temperature: float = 24,
    soil_npk: float = 72,
    irrigation_hours: float = 30,
    crop: str = "wheat",
    season: str = "kharif",
    farm_size_hectares: float = DEFAULT_FARM_SIZE_HA,
) -> dict:
    """Build a single ordered feature row (one-hot encoded) for inference."""
    crop_key = normalize_crop(crop)
    season_key = normalize_season(season)

    row = {
        "rainfall": float(rainfall),
        "temperature": float(temperature),
        "soil_npk": float(soil_npk),
        "irrigation_hours": float(irrigation_hours),
        "farm_size_hectares": float(farm_size_hectares),
    }
    for name in CROPS:
        row[f"crop_{name}"] = 1.0 if name == crop_key else 0.0
    for name in SEASONS:
        row[f"season_{name}"] = 1.0 if name == season_key else 0.0
    return row


def build_feature_frame(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Build a DataFrame with exactly ``FEATURE_COLUMNS`` in the right order.

    Only keys in :data:`INPUT_COLUMNS` are consumed, so callers may pass a
    full training row (including the target) without it leaking into the
    feature matrix.
    """
    rows = [
        build_feature_row(**{key: record[key] for key in INPUT_COLUMNS if key in record})  # type: ignore[arg-type]
        for record in records
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
