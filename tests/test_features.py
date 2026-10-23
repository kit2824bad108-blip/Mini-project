"""Unit tests for the shared feature-engineering module."""

import pandas as pd
import pytest

from features import (
    CROPS,
    FEATURE_COLUMNS,
    SEASONS,
    build_feature_frame,
    build_feature_row,
    heuristic_yield,
    normalize_crop,
    normalize_season,
)


class TestNormalization:
    @pytest.mark.parametrize("crop", CROPS)
    def test_valid_crops_are_returned_unchanged(self, crop):
        assert normalize_crop(crop) == crop

    @pytest.mark.parametrize("season", SEASONS)
    def test_valid_seasons_are_returned_unchanged(self, season):
        assert normalize_season(season) == season

    def test_crop_and_season_are_case_insensitive(self):
        assert normalize_crop("  WHEAT ") == "wheat"
        assert normalize_season("Kharif") == "kharif"

    def test_unknown_crop_raises(self):
        with pytest.raises(ValueError, match="Unknown crop"):
            normalize_crop("banana")

    def test_unknown_season_raises(self):
        with pytest.raises(ValueError, match="Unknown season"):
            normalize_season("monsoon")


class TestHeuristicYield:
    def test_returns_positive_yield_for_baseline_inputs(self):
        assert heuristic_yield(950, 24, 72, 30, "wheat", "kharif") > 0

    def test_matches_documented_baseline(self):
        # 0.80 * 1.00 (wheat) * 1.08 (kharif) * 1.048 (24C) * 1.16625 (950mm)
        # * 1.288 (72 NPK) * 1.03 (30h irrigation)
        assert heuristic_yield(950, 24, 72, 30, "wheat", "kharif") == pytest.approx(1.4013, abs=5e-4)

    def test_more_rainfall_never_reduces_yield_until_saturation(self):
        dry = heuristic_yield(300, 24, 72, 30, "wheat", "kharif")
        wet = heuristic_yield(1200, 24, 72, 30, "wheat", "kharif")
        assert wet > dry

    def test_saturated_rainfall_is_capped(self):
        at_cap = heuristic_yield(2000, 24, 72, 30, "wheat", "kharif")
        beyond = heuristic_yield(9000, 24, 72, 30, "wheat", "kharif")
        assert beyond == pytest.approx(at_cap)

    def test_better_soil_increases_yield(self):
        assert heuristic_yield(950, 24, 90, 30, "wheat", "kharif") > heuristic_yield(
            950, 24, 20, 30, "wheat", "kharif"
        )

    def test_crop_ranking_is_respected(self):
        rice = heuristic_yield(950, 24, 72, 30, "rice", "kharif")
        soybean = heuristic_yield(950, 24, 72, 30, "soybean", "kharif")
        assert rice > soybean

    def test_yield_is_never_negative(self):
        assert heuristic_yield(0, -5, 0, 0, "soybean", "rabi") >= 0


class TestFeatureFrame:
    def test_row_has_every_feature_column(self):
        row = build_feature_row()
        assert set(row) == set(FEATURE_COLUMNS)

    def test_one_hot_encoding_is_exclusive(self):
        row = build_feature_row(crop="maize", season="rabi")
        crop_flags = [row[f"crop_{c}"] for c in CROPS]
        season_flags = [row[f"season_{s}"] for s in SEASONS]

        assert sum(crop_flags) == 1
        assert sum(season_flags) == 1
        assert row["crop_maize"] == 1.0
        assert row["season_rabi"] == 1.0

    def test_frame_columns_are_in_canonical_order(self):
        frame = build_feature_frame([{}, {"crop": "rice", "season": "zaid"}])
        assert list(frame.columns) == FEATURE_COLUMNS
        assert len(frame) == 2

    def test_frame_rejects_unknown_crop(self):
        with pytest.raises(ValueError):
            build_feature_frame([{"crop": "quinoa"}])

    def test_frame_is_numeric_and_finite(self):
        frame = build_feature_frame([{}, {}])
        assert all(pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns)
        assert frame.notna().all().all()
