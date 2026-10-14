"""
Forecasting tests — mostly guarding against the two silent failures:
a baseline that peeks at the future, and metrics that lie on zero-inflated
series.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG                                   # noqa: E402
from src.models.forecast import (                             # noqa: E402
    climatology, load_panel, persistence, score, split_dates,
)


@pytest.fixture(scope="module")
def panel():
    path = Path(__file__).resolve().parents[1] / CFG.data.processed
    if not path.exists():
        pytest.skip("Run the pipeline first: ./scripts/run_pipeline.sh")
    return load_panel(CFG.forecast.target)


def test_persistence_predicts_the_present_not_the_future(panel):
    """y_pred at origin t must equal the observed value AT t, and y_true
    must be the value at t+h. Getting these backwards is a silent cheat."""
    target = CFG.forecast.target
    h = 3
    out = persistence(panel, target, h)

    district = out["district"].iloc[0]
    src = panel[panel["district"] == district].sort_values("date").reset_index(drop=True)
    got = out[out["district"] == district].sort_values("date").reset_index(drop=True)

    merged = got.merge(
        src.rename(columns={target: "observed_at_t"}), on=["district", "date"]
    )
    assert np.allclose(merged["y_pred"], merged["observed_at_t"])

    shifted = src.assign(date=src["date"] - pd.Timedelta(days=h))[["date", target]]
    merged2 = got.merge(shifted.rename(columns={target: "observed_at_t_plus_h"}),
                        on="date")
    assert np.allclose(merged2["y_true"], merged2["observed_at_t_plus_h"])


def test_climatology_uses_only_training_years(panel):
    """The climatology must be identical whether or not test rows exist."""
    target = CFG.forecast.target
    train_end, _ = split_dates()

    full = climatology(panel, target, 1)
    truncated = climatology(panel[panel["date"] <= train_end], target, 1)

    merged = full.merge(truncated, on=["district", "date"], suffixes=("_f", "_t"))
    assert len(merged) > 0
    # Predictions on overlapping dates must match exactly — if they differ,
    # the climatology absorbed information from the test period.
    assert np.allclose(merged["y_pred_f"], merged["y_pred_t"], equal_nan=True)


def test_mape_is_suppressed_on_zero_inflated_targets():
    y = np.array([0.0, 0.0, 4.0, 0.0, 9.0])
    yhat = np.array([0.5, 0.0, 3.0, 1.0, 8.0])

    disallowed = score(y, yhat, allow_mape=False)
    assert np.isnan(disallowed["mape"]), "MAPE must not be reported on zero-inflated data"
    assert not np.isnan(disallowed["smape"])
    assert disallowed["mae"] > 0


def test_metrics_ignore_nan_pairs():
    y = np.array([1.0, np.nan, 3.0])
    yhat = np.array([1.0, 2.0, np.nan])
    s = score(y, yhat, allow_mape=True)
    assert s["n"] == 1
    assert s["mae"] == pytest.approx(0.0)


def test_skill_score_sign_convention():
    """skill = 1 - mae_model/mae_persistence; positive means better."""
    mae_persistence, mae_better, mae_worse = 1.0, 0.8, 1.25
    assert 1 - mae_better / mae_persistence > 0
    assert 1 - mae_worse / mae_persistence < 0
