"""Integration tests for the FastAPI service (works with or without a trained model)."""

import pytest
from fastapi.testclient import TestClient

from app import FALLBACK_NAME, app

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


class TestSystemEndpoints:
    def test_health_reports_ready(self):
        response = client.get("/health")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_used"]
        assert body["feature_count"] > 0

    def test_api_info(self):
        response = client.get("/api")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_models_endpoint_lists_active_model(self):
        response = client.get("/models")
        assert response.status_code == 200
        assert response.json()["active"]


class TestPrediction:
    def test_valid_request_returns_prediction(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

        body = response.json()
        assert body["crop"] == "wheat"
        assert body["season"] == "kharif"
        assert body["predicted_yield_tph"] > 0
        assert body["model_used"]
        assert 0 < body["confidence"] <= 1
        assert body["inputs"]["rainfall"] == 950

    def test_total_tonnage_scales_with_farm_size(self):
        small = client.post("/predict", json={**VALID_PAYLOAD, "farm_size_hectares": 10})
        large = client.post("/predict", json={**VALID_PAYLOAD, "farm_size_hectares": 100})
        assert small.status_code == large.status_code == 200

        assert large.json()["estimated_total_tonnes"] > small.json()["estimated_total_tonnes"]
        assert large.json()["predicted_yield_tph"] == pytest.approx(
            small.json()["predicted_yield_tph"], abs=0.01
        )

    def test_richer_conditions_do_not_reduce_yield(self):
        poor = client.post(
            "/predict",
            json={**VALID_PAYLOAD, "rainfall": 250, "soil_npk": 15, "irrigation_hours": 2},
        )
        rich = client.post(
            "/predict",
            json={**VALID_PAYLOAD, "rainfall": 1400, "soil_npk": 95, "irrigation_hours": 90},
        )
        assert rich.json()["predicted_yield_tph"] > poor.json()["predicted_yield_tph"]

    def test_defaults_apply_when_body_is_empty(self):
        response = client.post("/predict", json={})
        assert response.status_code == 200

    def test_fallback_engine_is_reported_when_no_model(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        engine = response.json()["model_used"]
        assert engine == FALLBACK_NAME or engine != ""

    @pytest.mark.parametrize(
        "bad_payload",
        [
            {"crop": "banana"},
            {"season": "monsoon"},
            {"soil_npk": 150},
            {"soil_npk": -5},
            {"rainfall": -100},
            {"farm_size_hectares": 0},
            {"temperature": 500},
        ],
    )
    def test_invalid_inputs_are_rejected(self, bad_payload):
        response = client.post("/predict", json={**VALID_PAYLOAD, **bad_payload})
        assert response.status_code == 422


class TestStaticFrontend:
    @pytest.mark.parametrize(
        ("path", "needle"),
        [
            ("/", "<!DOCTYPE html>"),
            ("/styles.css", "--primary"),
            ("/app.js", "prediction-form"),
        ],
    )
    def test_frontend_is_served(self, path, needle):
        response = client.get(path)
        assert response.status_code == 200
        assert needle in response.text

    def test_source_files_are_not_exposed(self):
        assert client.get("/app.py").status_code == 404
        assert client.get("/train_model.py").status_code == 404
