"""
Feature reference statistics — fitted at training time, reused at serving time.

WHY THIS FILE EXISTS
--------------------
Several features are not functions of today's observation alone. They are
functions of today's observation *relative to a reference distribution*:

  rainfall_departure_mm   today minus the day-of-year climatology
  *_pctile                where today sits in this district's own history
  (NaN fill)              the training-period median of each feature

During training those references are derived from the training split. If
live scoring re-derives them from whatever window it happens to have, the
same physical rainfall produces a different feature value, and the model is
being fed something it was never trained on. That is train/serve skew, and
it is the most common way a model that looked fine offline quietly produces
nonsense in production.

So the references are fitted once, saved alongside the model, and loaded
back for every live prediction. If the reference file is missing, live
scoring refuses to run rather than silently improvising one.

WHAT IS STORED
  climatology    day-of-year mean rainfall per district (smoothed)
  quantiles      a 201-point quantile grid per (district, column), which is
                 a compact stand-in for the full training sample
  medians        per-feature training median, for NaN fill
  features       the exact feature order the model expects
  fitted_on      the training window, for provenance
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG, ROOT

# Columns that get a percentile-rank feature.
PCTILE_COLUMNS = ("rainfall_mm", "river_discharge_m3s", "antecedent_rain_mm")

# Resolution of the stored quantile grid. 201 points is 0.5% resolution,
# which is finer than the signal and far smaller than keeping the raw sample.
QUANTILE_POINTS = 201


@dataclass
class FeatureReference:
    climatology: pd.DataFrame
    quantiles: dict[tuple[str, str], np.ndarray]
    medians: pd.Series
    features: list[str] = field(default_factory=list)
    fitted_on: dict = field(default_factory=dict)

    # ---- percentile lookup -------------------------------------------
    def percentile(self, district: str, column: str, values: np.ndarray) -> np.ndarray:
        grid = self.quantiles.get((district, column))
        if grid is None:
            return np.full(len(values), np.nan)
        ranks = np.searchsorted(grid, np.asarray(values, dtype=float), side="right")
        return ranks / len(grid)

    # ---- persistence --------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: Path | None = None) -> "FeatureReference":
        path = path or default_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Feature reference not found at {path}.\n"
                f"It is written when features are built. Run:\n"
                f"    python -m src.features.build_features --horizon 1\n"
                f"Live scoring will not improvise a reference — that would "
                f"feed the model differently-scaled features than it was "
                f"trained on."
            )
        with open(path, "rb") as fh:
            return pickle.load(fh)


def default_path() -> Path:
    return ROOT / CFG.outputs.models / "feature_reference.pkl"


def fit(df: pd.DataFrame, train_end: pd.Timestamp,
        climatology: pd.DataFrame, medians: pd.Series,
        features: list[str]) -> FeatureReference:
    """Fit the reference from the TRAINING portion of df only."""
    ref_rows = df[df["date"] <= train_end]

    quantiles: dict[tuple[str, str], np.ndarray] = {}
    probs = np.linspace(0.0, 1.0, QUANTILE_POINTS)
    for column in PCTILE_COLUMNS:
        if column not in ref_rows.columns:
            continue
        for district, grp in ref_rows.groupby("district"):
            sample = grp[column].dropna().to_numpy(dtype=float)
            if sample.size < 50:
                continue
            quantiles[(district, column)] = np.quantile(sample, probs)

    return FeatureReference(
        climatology=climatology,
        quantiles=quantiles,
        medians=medians,
        features=list(features),
        fitted_on={
            "train_end": str(train_end.date()),
            "n_rows": int(len(ref_rows)),
            "districts": sorted(ref_rows["district"].unique().tolist()),
            "source": str(CFG.data.source),
        },
    )
