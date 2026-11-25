"""
Probability calibration.

WHY THIS FILE EXISTS
--------------------
A classifier trained with `class_weight="balanced"` (or XGBoost's
`scale_pos_weight`) is deliberately told that the rare class matters more
than its frequency implies. That is the right thing to do for ranking and
for recall, and it is the reason the models detect floods at all at a 1.6
per cent base rate. It also guarantees that the number coming out of
`predict_proba` is NOT a probability: reweighting shifts the decision
boundary and inflates the score. Measured on the test split before this
module existed, mean predicted risk was 0.269 for logistic regression and
0.112 for the random forest against an actual onset rate of 0.016 — an
over-statement of roughly 16x and 7x. Both scored worse on the Brier score
than a model that always predicts zero.

That matters operationally, not just cosmetically. A district officer who
reads "62 per cent" needs it to mean that floods follow about 62 per cent
of such warnings. An uncalibrated score cannot be read that way, so the
only thing it can be used for is ranking districts against each other.

WHAT IS DONE ABOUT IT
---------------------
Each fitted model gets a monotone post-hoc calibration map, fitted on
predictions the model never trained on, and applied before any threshold
is chosen or any number is shown.

  - isotonic  (default) a non-parametric monotone step function. It can
              correct an arbitrary monotone distortion, which is what
              reweighting produces, and needs a few hundred positives to
              be stable. It cannot change the RANKING, so ROC-AUC and
              PR-AUC are untouched by construction — only Brier,
              reliability and the meaning of the threshold change.
  - sigmoid   Platt scaling: a one-parameter logistic fit on the logit of
              the score. Smoother, more stable on small samples, but it
              can only correct a logistic-shaped distortion.
  - none      pass-through, for ablation.

WHERE THE CALIBRATION DATA COMES FROM
-------------------------------------
Not the validation split. Validation is already spent on choosing the
alert threshold, and fitting the calibrator on the same rows that pick the
threshold makes the operating point look better than it is.

Instead the calibration set is built from the TRAINING period by
`chronological_oof`: the training span is cut into equal date blocks, and
each block is predicted by a model fitted only on the blocks before it.
Every calibration point is therefore out-of-sample for the model that
produced it, no future data is used to predict the past, and the
validation and test splits are left untouched. The cost is one extra model
fit per block.

Usage
-----
    from src.models.calibrate import chronological_oof, fit_calibrator

    p_oof, y_oof = chronological_oof(estimator, train_df, feats, TARGET)
    cal = fit_calibrator(p_oof, y_oof, method="isotonic")
    calibrated = cal.transform(model.predict_proba(X)[:, 1])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


class Calibrator:
    """A fitted, monotone score -> probability map.

    Pickled alongside the model, so a live prediction applies exactly the
    map that was fitted at training time. `transform` is the only thing
    callers need; the rest is provenance for the report.
    """

    def __init__(self, method: str, mapper=None, n_fit: int = 0,
                 base_rate: float = float("nan"),
                 mean_in: float = float("nan"),
                 mean_out: float = float("nan")):
        self.method = method
        self.mapper = mapper
        self.n_fit = int(n_fit)
        self.base_rate = float(base_rate)
        self.mean_in = float(mean_in)
        self.mean_out = float(mean_out)

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none" or self.mapper is None:
            return p
        if self.method == "isotonic":
            out = self.mapper.predict(p)
        else:  # sigmoid / Platt
            out = self.mapper.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.clip(out, 0.0, 1.0)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"Calibrator(method={self.method!r}, n_fit={self.n_fit:,}, "
                f"mean {self.mean_in:.3f} -> {self.mean_out:.3f}, "
                f"base rate {self.base_rate:.4f})")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibrator(p, y, method: str = "isotonic") -> Calibrator:
    """Fit a monotone map from raw score to observed frequency."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)

    if method == "none" or len(p) == 0:
        return Calibrator("none", None, len(p), float(y.mean()) if len(y) else float("nan"),
                          float(p.mean()) if len(p) else float("nan"),
                          float(p.mean()) if len(p) else float("nan"))

    if y.sum() < 10:
        # Too few positives to fit anything trustworthy. Say so rather than
        # shipping a map fitted on noise.
        print(f"[calibrate] only {int(y.sum())} positives in the calibration "
              f"set — falling back to no calibration")
        return Calibrator("none", None, len(p), float(y.mean()),
                          float(p.mean()), float(p.mean()))

    if method == "isotonic":
        mapper = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        mapper.fit(p, y)
    elif method == "sigmoid":
        mapper = LogisticRegression(max_iter=1000)
        mapper.fit(_logit(p).reshape(-1, 1), y)
    else:
        raise ValueError(f"unknown calibration method {method!r} "
                         f"(expected isotonic, sigmoid or none)")

    cal = Calibrator(method, mapper, len(p), float(y.mean()), float(p.mean()))
    cal.mean_out = float(cal.transform(p).mean())
    return cal


def chronological_oof(estimator, df: pd.DataFrame, feats: list[str],
                      target: str, blocks: int = 4,
                      rebalance: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-sample predictions over the training period, past -> future only.

    The training span is cut into `blocks` equal-length date blocks. Block
    i is predicted by a fresh clone of `estimator` fitted on blocks 0..i-1,
    so the first block has no prediction and the remaining (blocks - 1)/blocks
    of the training rows do. Expanding-window rather than sliding, because
    more history is strictly better here and the alternative discards it.

    `rebalance` recomputes scale_pos_weight per fold for estimators that
    take it, so each fold sees the same imbalance correction the final model
    will.
    """
    work = df.sort_values("date")
    # Equal-count date blocks. Quantiles of the date column rather than a
    # fixed calendar split, so each fold sees a comparable number of rows
    # however unevenly the districts are covered.
    qs = (pd.Series(work["date"].to_numpy())
          .quantile(np.linspace(0, 1, blocks + 1)).to_numpy())
    edges = pd.to_datetime(qs)

    p_parts, y_parts = [], []
    for i in range(1, blocks):
        fit_mask = work["date"] < edges[i]
        pred_mask = (work["date"] >= edges[i]) & (work["date"] < edges[i + 1])
        if i == blocks - 1:
            pred_mask = work["date"] >= edges[i]

        fit_rows, pred_rows = work[fit_mask], work[pred_mask]
        y_fit = fit_rows[target]
        if pred_rows.empty or y_fit.nunique() < 2:
            continue

        est = clone(estimator)
        if rebalance and "scale_pos_weight" in est.get_params():
            est.set_params(scale_pos_weight=float(
                (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)))

        est.fit(fit_rows[feats], y_fit)
        p_parts.append(est.predict_proba(pred_rows[feats])[:, 1])
        y_parts.append(pred_rows[target].to_numpy())
        print(f"[calibrate]   fold {i}/{blocks - 1}: fit {len(fit_rows):,} rows "
              f"-> predict {len(pred_rows):,} rows "
              f"({int(y_parts[-1].sum())} positives)")

    if not p_parts:
        return np.array([]), np.array([])
    return np.concatenate(p_parts), np.concatenate(y_parts)
