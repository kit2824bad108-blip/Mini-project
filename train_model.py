"""
train_model.py
==============
Train, evaluate, and persist crop-yield regression models.

The script synthesises a representative farm dataset, fits several regression
models, evaluates them (R2 / MAE / RMSE), then saves the best performer along
with the feature schema and metrics so the inference API can reload everything.

Usage:
    python train_model.py
    python train_model.py --n-samples 20000 --models xgboost
    python train_model.py --output models/best_xgboost.pkl

The saved artifacts are:
    models/best_xgboost.pkl      fitted scikit-learn compatible estimator
    models/feature_columns.json  ordered feature schema for inference
    models/metrics.json          per-model evaluation metrics
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from features import (
    CROPS,
    FEATURE_COLUMNS,
    SEASONS,
    build_feature_frame,
    heuristic_yield,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.2

#: Model registry — keys are the values accepted by --models.
MODEL_REGISTRY = {
    "random_forest": lambda seed: RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=seed, n_jobs=-1,
    ),
    "xgboost": lambda seed: _xgboost_regressor(seed),
    "svr": lambda seed: SVR(kernel="rbf", C=100, epsilon=0.1),
    "dnn": lambda seed: MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=500,
        random_state=seed,
    ),
}

MODEL_ORDER = ["random_forest", "xgboost", "svr", "dnn"]


def _xgboost_regressor(seed: int):
    """Import XGBoost lazily so the module loads without it."""
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_samples: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic dataset of historical farm records.

    The target is produced by the same reference model used offline by the API
    (``features.heuristic_yield``) plus small measurement noise, so a model
    that learns well should approach the noise ceiling.
    """
    rng = np.random.default_rng(seed)
    n = int(n_samples)

    print(f"Generating {n:,} synthetic farm records (seed={seed})...")

    df = pd.DataFrame(
        {
            "rainfall": rng.uniform(200, 1800, n),
            "temperature": rng.uniform(15, 35, n),
            "soil_npk": rng.uniform(0, 100, n),
            "irrigation_hours": rng.uniform(0, 150, n),
            "farm_size_hectares": rng.uniform(1, 200, n),
            "crop": rng.choice(CROPS, n, p=[0.25, 0.30, 0.25, 0.20]),
            "season": rng.choice(SEASONS, n, p=[0.30, 0.40, 0.30]),
        }
    )

    expected = df.apply(
        lambda row: heuristic_yield(
            row["rainfall"],
            row["temperature"],
            row["soil_npk"],
            row["irrigation_hours"],
            row["crop"],
            row["season"],
        ),
        axis=1,
    )

    df["yield_tph"] = np.maximum(expected + rng.normal(0, 0.05, n), 0.1)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_names: list[str],
    seed: int = RANDOM_STATE,
):
    """Fit each requested model and return its metrics plus the fitted objects."""
    results: list[dict] = []
    fitted: dict[str, object] = {}

    print("\n" + "=" * 68)
    print("Training and evaluating models")
    print("=" * 68)

    for name in model_names:
        print(f"\n>>> Training {name} ...")
        model = MODEL_REGISTRY[name](seed)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = r2_score(y_test, predictions)

        results.append(
            {"model": name, "r2": round(float(r2), 4), "mae": round(float(mae), 4),
             "rmse": round(rmse, 4)}
        )
        fitted[name] = model
        print(f"    R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    return pd.DataFrame(results), fitted


def save_artifacts(model, metrics_df: pd.DataFrame, output: Path) -> Path:
    """Persist the fitted model, the feature schema and the metrics."""
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as handle:
        pickle.dump(model, handle)

    schema_path = output.parent / "feature_columns.json"
    schema_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")

    metrics_path = output.parent / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "models": metrics_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[OK] Model saved          -> {output}")
    print(f"[OK] Feature schema saved -> {schema_path}")
    print(f"[OK] Metrics saved        -> {metrics_path}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate crop-yield regression models."
    )
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of synthetic training samples (default: 10000)")
    parser.add_argument("--output", type=str, default="models/best_xgboost.pkl",
                        help="Where to write the best model (default: models/best_xgboost.pkl)")
    parser.add_argument("--models", type=str, default=",".join(MODEL_ORDER),
                        help=f"Comma-separated subset of: {', '.join(MODEL_ORDER)}")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help=f"Random seed (default: {RANDOM_STATE})")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    requested = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in requested if name not in MODEL_REGISTRY]
    if unknown:
        print(f"error: unknown model(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"       choose from: {', '.join(MODEL_ORDER)}", file=sys.stderr)
        return 2

    print("=" * 68)
    print("Crop Yield Prediction - Model Training")
    print("=" * 68)

    df = generate_synthetic_data(args.n_samples, args.seed)
    X = build_feature_frame(df.to_dict(orient="records"))
    y = df["yield_tph"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=args.seed
    )

    metrics_df, fitted = train_and_evaluate(
        X_train, X_test, y_train, y_test, requested, args.seed
    )

    print("\n" + "=" * 68)
    print("Model comparison")
    print("=" * 68)
    print(metrics_df.to_string(index=False))

    best_row = metrics_df.loc[metrics_df["r2"].idxmax()]
    best_name = str(best_row["model"])
    print(f"\nBest model: {best_name} (R2 = {best_row['r2']:.4f})")

    save_artifacts(fitted[best_name], metrics_df, Path(args.output))
    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
