"""
Tests for the live/forward-looking path.

The backtest is protected by tests/test_no_leakage.py. This file protects
the other failure mode: the model being fed something at SERVING time that
it never saw at TRAINING time. That bug does not show up in any offline
metric — the backtest stays beautiful while the live page quietly produces
nonsense — so it has to be asserted rather than eyeballed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.reference import FeatureReference, default_path  # noqa: E402
from src.live import predict as lp  # noqa: E402

DISTRICTS = ["Wayanad", "Cachar", "Patna"]


def _synthetic_conditions(days: int = 120) -> pd.DataFrame:
    """A live-shaped feed, built locally so the test needs no network.

    Same columns and dtypes src.live.fetch writes: a run of observed days
    followed by a forecast tail.
    """
    rng = np.random.default_rng(7)
    end = pd.Timestamp("2024-07-15")
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    cutoff = dates[-10]

    rows = []
    for name in DISTRICTS:
        rows.append(pd.DataFrame({
            "date": dates,
            "district": name,
            "rainfall_mm": rng.gamma(2.0, 6.0, len(dates)),
            "river_discharge_m3s": rng.gamma(3.0, 200.0, len(dates)),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["is_forecast"] = df["date"] > cutoff
    return df.sort_values(["district", "date"]).reset_index(drop=True)


def test_terrain_falls_back_to_the_committed_dataset():
    """A deploy has no data/raw/. Live scoring must still find terrain.

    data/raw/ is gitignored, so on a hosted copy of this repo the raw
    elevation CSV does not exist. If terrain resolution depended on it, the
    whole live tab would be local-only.
    """
    table = lp.terrain_table(DISTRICTS, prefer_raw=False)
    assert set(lp.TERRAIN_COLUMNS).issubset(table.columns)
    assert set(table["district"]) <= set(DISTRICTS)
    assert not table.empty, "the committed dataset carries no terrain columns"
    assert table["district"].is_unique
    assert table["mean_elevation_m"].notna().all()


def test_terrain_sources_agree():
    """Whichever source is used, the numbers must be the same ones.

    If the raw download and the committed dataset disagreed about a
    district's elevation, the live page and the backtest would be running
    different models in effect.
    """
    raw = ROOT / "data/raw/terrain_elevation.csv"
    if not raw.exists():
        pytest.skip("raw terrain download not present (expected on a deploy)")

    a = lp.terrain_table(DISTRICTS, prefer_raw=True).set_index("district")
    b = lp.terrain_table(DISTRICTS, prefer_raw=False).set_index("district")
    shared = sorted(set(a.index) & set(b.index))
    assert shared, "the two terrain sources share no districts"
    for col in ["mean_elevation_m", "mean_slope_deg", "lat", "lon"]:
        np.testing.assert_allclose(
            a.loc[shared, col].to_numpy(float),
            b.loc[shared, col].to_numpy(float),
            rtol=1e-6, atol=1e-6,
            err_msg=f"terrain sources disagree on {col}",
        )


def test_percentile_lookup_is_monotonic_and_bounded():
    """The percentile feature has to behave like a percentile."""
    grid = {("X", "rainfall_mm"): np.linspace(0.0, 100.0, 201)}
    ref = FeatureReference(climatology=pd.DataFrame(), quantiles=grid,
                           medians=pd.Series(dtype=float))
    out = ref.percentile("X", "rainfall_mm", np.array([0.0, 25.0, 50.0, 200.0]))
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert np.all(np.diff(out) >= 0), "percentile must be non-decreasing"
    # An unknown district has no grid, and must say so rather than guess 0.
    assert np.isnan(ref.percentile("ZZ", "rainfall_mm", np.array([1.0]))).all()


def test_live_features_cover_every_feature_the_model_expects():
    """The train/serve skew guard.

    Build features over a live-shaped window using the SAVED reference, and
    assert that every column the trained model asks for exists and is
    finite. A missing or NaN feature here is the exact bug this whole
    reference mechanism exists to prevent.
    """
    if not default_path().exists():
        pytest.skip("no feature reference on disk — run build_features first")

    ref = FeatureReference.load()
    built = lp.build_live_features(_synthetic_conditions(), ref)

    missing = [c for c in ref.features if c not in built.columns]
    assert not missing, f"live build is missing {missing[:8]}"

    tail = built[built["date"] == built["date"].max()]
    values = tail[ref.features].to_numpy(dtype=float)
    assert np.isfinite(values).all(), (
        "non-finite feature values at the scoring row: "
        f"{[c for c in ref.features if not np.isfinite(tail[c].to_numpy(float)).all()][:8]}"
    )


def test_reference_is_fitted_on_training_data_only():
    """The reference must not have seen validation or test rows.

    If it had, the percentile and departure features would carry
    information from the future — leakage dressed up as normalisation.
    """
    if not default_path().exists():
        pytest.skip("no feature reference on disk — run build_features first")

    from src.config import CFG

    ref = FeatureReference.load()
    assert ref.fitted_on.get("train_end") == str(CFG.split.train_end), (
        f"reference was fitted to {ref.fitted_on.get('train_end')} but the "
        f"configured training split ends {CFG.split.train_end}"
    )
