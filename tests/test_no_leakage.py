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


def test_ongoing_target_is_strictly_the_next_day(features):
    """The "ongoing" target at t must equal the raw label at t+horizon.

    The default target mode is onset_window, so this builds the "ongoing"
    target explicitly rather than reading the parquet — that path still
    exists and still has to be correct.
    """
    from src.features.build_features import _add_target

    raw_path = Path(__file__).resolve().parents[1] / CFG.data.processed
    raw = pd.read_parquet(raw_path)[["district", "date", "flood_occurred"]]
    built = _add_target(raw.copy(), horizon=1, mode="ongoing")

    merged = built[["district", "date", TARGET]].merge(
        raw.assign(date=raw["date"] - pd.Timedelta(days=1)),
        on=["district", "date"], how="inner",
    )
    assert len(merged) > 0
    assert (merged[TARGET] == merged["flood_occurred"]).all(), (
        "ongoing target is not the next day's label — the horizon shift is wrong."
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


def test_onset_target_looks_forward_only(features):
    """The onset-window target must be built from FUTURE onsets only.

    A target that accidentally includes today's onset would hand the model
    the answer. Rebuild the target by brute force on one district and
    compare against what the vectorised forward-window produced.
    """
    import numpy as np
    from src.features.build_features import _add_target

    raw_path = Path(__file__).resolve().parents[1] / CFG.data.processed
    raw = pd.read_parquet(raw_path)[["district", "date", "flood_occurred"]]

    horizon, window = 1, int(getattr(CFG.features, "target_window", 7))
    built = _add_target(raw.copy(), horizon, mode="onset_window", window=window)

    d = built["district"].iloc[0]
    one = built[built["district"] == d].sort_values("date").reset_index(drop=True)
    flood = one["flood_occurred"].to_numpy()
    onset = np.zeros(len(flood), dtype=int)
    onset[0] = flood[0]
    onset[1:] = ((flood[1:] == 1) & (flood[:-1] == 0)).astype(int)

    expected = np.zeros(len(flood))
    for i in range(len(flood)):
        lo, hi = i + horizon, i + horizon + window
        seg = onset[lo:hi]
        expected[i] = seg.max() if len(seg) else np.nan

    got = one[TARGET].to_numpy(dtype=float)
    valid = ~np.isnan(expected) & ~np.isnan(got)
    assert valid.sum() > 100
    assert (expected[valid] == got[valid]).all(), (
        "onset-window target does not match a brute-force forward window"
    )


def test_district_identity_is_never_a_feature(features):
    """Basin may enter as a dummy; raw district identity must not."""
    from src.features.build_features import feature_columns
    feats = feature_columns(features)
    assert "district" not in feats
    assert not any(f.startswith("district_") for f in feats)
    assert "lat" not in feats and "lon" not in feats
