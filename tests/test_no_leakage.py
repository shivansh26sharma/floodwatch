"""
Leakage tests. These are the tests that protect your grade.

A flood model that scores 0.99 AUC is almost always leaking. Run these
after any change to feature engineering:

    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG                                      # noqa: E402
from src.features.build_features import (                        # noqa: E402
    TARGET, chronological_split, fit_climatology,
)


@pytest.fixture(scope="module")
def features():
    path = Path(CFG.data.processed).parent / "features_h1.parquet"
    full = Path(__file__).resolve().parents[1] / path
    if not full.exists():
        pytest.skip("Run the pipeline first: ./scripts/run_pipeline.sh")
    return pd.read_parquet(full)


def test_splits_are_chronological_and_disjoint(features):
    s = chronological_split(features)
    assert s["train"]["date"].max() < s["val"]["date"].min()
    assert s["val"]["date"].max() < s["test"]["date"].min()
    assert len(s["train"]) and len(s["val"]) and len(s["test"])


def test_no_overlapping_district_days_across_splits(features):
    s = chronological_split(features)
    keys = {
        name: set(zip(part["district"], part["date"]))
        for name, part in s.items()
    }
    assert not keys["train"] & keys["val"]
    assert not keys["val"] & keys["test"]
    assert not keys["train"] & keys["test"]


def test_target_is_strictly_future(features):
    """The target at date t must equal the raw label at t+horizon."""
    raw_path = Path(__file__).resolve().parents[1] / CFG.data.processed
    raw = pd.read_parquet(raw_path)[["district", "date", "flood_occurred"]]

    merged = features[["district", "date", TARGET]].merge(
        raw.assign(date=raw["date"] - pd.Timedelta(days=1)),
        on=["district", "date"], how="inner",
    )
    assert len(merged) > 0
    assert (merged[TARGET] == merged["flood_occurred"]).all(), (
        "Target is not the next day's label — the horizon shift is wrong."
    )


def test_current_label_is_not_a_feature(features):
    """flood_occurred (today) must never be fed to the model."""
    from src.features.build_features import feature_columns
    feats = feature_columns(features)
    assert "flood_occurred" not in feats
    assert TARGET not in feats


def test_climatology_is_fitted_on_train_only(features):
    """Refitting the climatology on test data must change it.

    If it does not, the climatology is not actually period-specific and
    the leak-guard is vacuous.
    """
    s = chronological_split(features)
    clim_train = fit_climatology(s["train"])
    clim_test = fit_climatology(s["test"])

    merged = clim_train.merge(
        clim_test, on=["district", "doy"], suffixes=("_train", "_test")
    )
    assert len(merged) > 0
    diff = (
        merged["rainfall_climatology_mm_train"]
        - merged["rainfall_climatology_mm_test"]
    ).abs().mean()
    assert diff > 1e-9


def test_no_rolling_feature_uses_the_current_day(features):
    """Rolling windows use closed='left', so a rolling mean over window w
    must not equal a window that includes today."""
    df = features.sort_values(["district", "date"])
    one = df[df["district"] == df["district"].iloc[0]].head(40)
    if "rainfall_mm_rollmean3" not in one.columns:
        pytest.skip("feature not present")

    inclusive = one["rainfall_mm"].rolling(3, min_periods=1).mean()
    exclusive = one["rainfall_mm_rollmean3"]
    # They must differ on at least some rows; identical means the window
    # is including the current day.
    assert (inclusive.values[5:] != exclusive.values[5:]).any()
