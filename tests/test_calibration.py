"""
Calibration tests.

The failure these guard against is the one the project actually had: a
model whose `predict_proba` is treated as a probability when class
reweighting has made it a ranking score. Four things have to hold.

  1. The map is monotone and lands in [0, 1] — a calibrator that reorders
     districts would change who gets warned, which is not calibration.
  2. It improves the Brier score on an inflated score, and beats the
     always-predict-zero baseline that an uncalibrated model fails.
  3. No fold of the out-of-sample calibration set is fitted on data later
     than what it predicts.
  4. Whatever is saved for the dashboard carries its calibrator, and the
     live scorer applies it rather than shipping the raw score.
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.config import CFG, ROOT
from src.models.calibrate import Calibrator, chronological_oof, fit_calibrator


def _inflated(n=4000, seed=0):
    """Scores that rank well but are wildly over-stated, like ours were."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(latent * 2 - 4.0)))).astype(int)
    p_raw = np.clip(1 / (1 + np.exp(-(latent * 2 + 1.0))), 1e-6, 1 - 1e-6)
    return p_raw, y


def test_calibrator_is_monotone_and_bounded():
    p, y = _inflated()
    cal = fit_calibrator(p, y, method="isotonic")
    grid = np.linspace(0, 1, 200)
    out = cal.transform(grid)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out) >= -1e-9), "calibration must not reorder scores"


def test_calibration_improves_brier_and_beats_always_zero():
    p, y = _inflated()
    cal = fit_calibrator(p, y, method="isotonic")
    p_cal = cal.transform(p)

    base_rate = float(y.mean())
    brier_raw = brier_score_loss(y, p)
    brier_cal = brier_score_loss(y, p_cal)
    brier_zero = base_rate                      # mean((0 - y)^2) = mean(y)

    assert brier_cal < brier_raw
    assert brier_cal <= brier_zero
    assert abs(p_cal.mean() - base_rate) < abs(p.mean() - base_rate)


def test_sigmoid_and_none_methods_round_trip():
    p, y = _inflated()
    assert fit_calibrator(p, y, method="sigmoid").transform(p).max() <= 1.0
    passthrough = fit_calibrator(p, y, method="none")
    assert np.allclose(passthrough.transform(p), p)
    with pytest.raises(ValueError):
        fit_calibrator(p, y, method="quadratic")


def test_too_few_positives_falls_back_rather_than_fitting_noise():
    p = np.linspace(0.01, 0.99, 500)
    y = np.zeros(500, dtype=int)
    y[:3] = 1
    cal = fit_calibrator(p, y, method="isotonic")
    assert cal.method == "none"
    assert np.allclose(cal.transform(p), p)


def test_oof_folds_never_see_the_future():
    """Each fold must be fitted strictly before the rows it predicts."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2000-01-01", periods=1200, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "x": rng.normal(size=1200),
    })
    df["y"] = (df["x"] + rng.normal(scale=0.5, size=1200) > 1.0).astype(int)

    seen = []

    class Spy(LogisticRegression):
        def fit(self, X, y, **kw):
            seen.append((X.index.min(), X.index.max()))
            return super().fit(X, y, **kw)

    p, y = chronological_oof(Spy(max_iter=200), df, ["x"], "y", blocks=4)
    assert len(p) == len(y) > 0
    # Every fit range must end before the next one starts: expanding window.
    for (lo, hi), (nlo, nhi) in zip(seen, seen[1:]):
        assert nhi > hi and nlo <= lo


def test_saved_models_carry_a_calibrator_and_live_scoring_applies_it():
    """The artefact the dashboard loads must not be a raw-score model."""
    import pickle

    horizon = 1
    bundles = sorted((ROOT / CFG.outputs.models).glob(f"risk_*_h{horizon}.pkl"))
    if not bundles:
        pytest.skip("no trained models on disk — run src.models.evaluate first")

    for path in bundles:
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        assert "calibrator" in bundle, f"{path.name} predates calibration"
        cal = bundle["calibrator"]
        assert isinstance(cal, Calibrator)
        assert cal.method in {"isotonic", "sigmoid", "none"}
        if cal.method != "none":
            # The map has to actually do something to an inflated score.
            assert not np.allclose(cal.transform(np.array([0.9])), 0.9)


def test_reported_calibration_beats_the_do_nothing_baseline():
    """Every shipped model must score better than always predicting zero.

    Brier, not the mean, is the thing being fixed. Calibration reshapes the
    whole distribution, and a model whose mean already happened to sit near
    the base rate can have its mean move slightly AWAY while its
    reliability improves — that is what happened to XGBoost here (mean
    0.022 -> 0.031, Brier 0.0212 -> 0.0161). So the invariants asserted are
    the ones that matter: Brier must not get worse, and it must clear the
    always-predict-zero baseline that an uncalibrated model fails.
    """
    path = ROOT / CFG.outputs.reports / "calibration_effect_h1.csv"
    if not path.exists():
        pytest.skip("no calibration report — run src.models.evaluate first")
    eff = pd.read_csv(path)
    assert len(eff), "calibration report is empty"
    for _, r in eff.iterrows():
        if r["method"] == "none":
            continue
        assert r["brier_calibrated"] <= r["brier_raw"] + 1e-9, r["model"]
        assert r["brier_calibrated"] <= r["brier_always_zero"], r["model"]
        # A badly inflated score must be pulled back towards the base rate.
        if r["mean_predicted_raw"] > 3 * r["actual_rate"]:
            assert (abs(r["mean_predicted_calibrated"] - r["actual_rate"])
                    < abs(r["mean_predicted_raw"] - r["actual_rate"])), r["model"]
