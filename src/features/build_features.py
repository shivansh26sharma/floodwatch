"""
Feature engineering.

THE MOST IMPORTANT FILE IN THE REPO FOR CORRECTNESS.
----------------------------------------------------
Every bug that makes a flood model look brilliant and be useless lives
here. Two rules are enforced, not just hoped for:

  RULE 1 — NO FUTURE INFORMATION.
      Every feature for district d at date t is computed from
      observations at dates <= t for district d only. All rolling and
      lag operations are grouped by district and use closed="left"
      where the current day must be excluded.

  RULE 2 — WE PREDICT FORWARD, NOT SIDEWAYS.
      The target is whether a flood occurs at t + horizon, using
      features available at t. If you instead predict the flood at t
      from the rainfall at t, you get a beautiful ROC curve and a
      system with zero warning time, which is worthless. The default
      horizon is 1 day; `python -m src.features.build_features --horizon 3`
      builds a 3-day-ahead dataset.

Climatology note: rainfall departure is computed against a day-of-year
climatology built from the TRAINING PERIOD ONLY and passed in, so the
test years cannot leak into the climatology. See `fit_climatology`.
"""

import argparse

import numpy as np
import pandas as pd

from src.config import CFG, ROOT

TARGET = "flood_occurred_future"
ID_COLS = ["district", "date"]
# Columns never fed to a model
NON_FEATURES = {
    "flood_occurred", TARGET, "district", "date", "lat", "lon",
    "rainfall_mm_was_missing", "river_discharge_m3s_was_missing",
}


# ----------------------------------------------------------------------
# Climatology (fit on train only, then applied everywhere)
# ----------------------------------------------------------------------

def fit_climatology(train: pd.DataFrame) -> pd.DataFrame:
    """Mean rainfall per (district, day-of-year), smoothed over a window.

    Fitted on the training split only. Passing test years in here is a
    subtle and very common leak.
    """
    df = train.copy()
    df["doy"] = df["date"].dt.dayofyear
    clim = (
        df.groupby(["district", "doy"], as_index=False)["rainfall_mm"]
        .mean()
        .rename(columns={"rainfall_mm": "rainfall_climatology_mm"})
    )
    # Smooth across neighbouring days-of-year so the climatology isn't spiky.
    clim = clim.sort_values(["district", "doy"])
    clim["rainfall_climatology_mm"] = (
        clim.groupby("district", group_keys=False)["rainfall_climatology_mm"]
        .apply(lambda s: s.rolling(15, center=True, min_periods=1).mean())
    )
    return clim


# ----------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------

def _lag_and_roll(df: pd.DataFrame) -> pd.DataFrame:
    cfg = CFG.features
    df = df.sort_values(ID_COLS).copy()
    g = df.groupby("district", group_keys=False)

    for col in ("rainfall_mm", "river_discharge_m3s"):
        for lag in cfg.lags:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)

        for window in cfg.rolling_windows:
            # closed="left" excludes the current day: the window covers
            # [t-window, t-1]. Combined with the current-day value being
            # available separately, nothing from the future is used.
            roll = g[col].rolling(window, min_periods=max(2, window // 3), closed="left")
            df[f"{col}_rollmean{window}"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{col}_rollmax{window}"] = roll.max().reset_index(level=0, drop=True)
            if col == "rainfall_mm":
                df[f"{col}_rollsum{window}"] = roll.sum().reset_index(level=0, drop=True)

    # Antecedent wetness: how saturated is the ground already?
    # This is the interaction that makes the same rainfall flood or not.
    w = cfg.antecedent_window
    df["antecedent_rain_mm"] = (
        g["rainfall_mm"].rolling(w, min_periods=3, closed="left").sum()
        .reset_index(level=0, drop=True)
    )

    # Rate of change — a rising river is more dangerous than a high static one.
    df["river_level_delta_1d"] = g["river_discharge_m3s"].diff(1)
    df["river_level_delta_3d"] = g["river_discharge_m3s"].diff(3)

    return df


def _add_departure(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear
    df = df.merge(clim, on=["district", "doy"], how="left")
    df["rainfall_departure_mm"] = df["rainfall_mm"] - df["rainfall_climatology_mm"]
    df["rainfall_departure_ratio"] = (
        df["rainfall_mm"] / df["rainfall_climatology_mm"].replace(0, np.nan)
    )
    return df.drop(columns=["doy"])


def _add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    doy = df["date"].dt.dayofyear
    # Cyclical encoding so December and January are adjacent.
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_monsoon"] = df["date"].dt.month.isin([6, 7, 8, 9]).astype(int)
    return df


def _add_terrain_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Physically motivated: the same rainfall is more dangerous on flat,
    # low-lying, well-drained-into terrain.
    df["rain_x_inv_slope"] = df["rainfall_mm"] / (df["mean_slope_deg"] + 1.0)
    df["antecedent_x_ruggedness"] = df["antecedent_rain_mm"] * df["terrain_ruggedness_m"]
    df["elevation_inv"] = 1.0 / (df["mean_elevation_m"] + 10.0)
    return df


def _add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Target = flood at t + horizon. Shift is per district, negative = future."""
    df = df.sort_values(ID_COLS).copy()
    df[TARGET] = (
        df.groupby("district", group_keys=False)["flood_occurred"].shift(-horizon)
    )
    return df


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------

def chronological_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Strictly time-ordered train / val / test. Never shuffle."""
    train_end = pd.Timestamp(CFG.split.train_end)
    val_end = pd.Timestamp(CFG.split.val_end)

    splits = {
        "train": df[df["date"] <= train_end],
        "val": df[(df["date"] > train_end) & (df["date"] <= val_end)],
        "test": df[df["date"] > val_end],
    }

    # Assert the splits really are disjoint and ordered.
    assert splits["train"]["date"].max() < splits["val"]["date"].min()
    assert splits["val"]["date"].max() < splits["test"]["date"].min()
    return splits


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])
    ]


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def build(horizon: int = 1, save: bool = True) -> tuple[pd.DataFrame, list[str]]:
    path = ROOT / CFG.data.processed
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Build it first:\n    python -m src.data.build_dataset"
        )
    raw = pd.read_parquet(path)

    # Climatology must be fitted on the training period only.
    train_mask = raw["date"] <= pd.Timestamp(CFG.split.train_end)
    clim = fit_climatology(raw[train_mask])

    df = _lag_and_roll(raw)
    df = _add_departure(df, clim)
    df = _add_seasonality(df)
    df = _add_terrain_interactions(df)
    df = _add_target(df, horizon)

    # Rows at the start of each district's series (no lag history) and at
    # the very end (no future label) are unusable.
    before = len(df)
    df = df.dropna(subset=[TARGET])
    df = df.dropna(thresh=int(0.9 * df.shape[1]))
    df[TARGET] = df[TARGET].astype(int)

    feats = feature_columns(df)
    # Any remaining NaNs in features are filled with the median computed on
    # the training period only — again, no test leakage. The mask is
    # recomputed from the date column rather than reused from `raw`,
    # because rows have been dropped and index alignment is fragile.
    df_train_mask = df["date"] <= pd.Timestamp(CFG.split.train_end)
    train_medians = df.loc[df_train_mask, feats].median()
    df[feats] = df[feats].fillna(train_medians)

    print(f"[features] {len(feats)} features, {len(df):,} usable rows "
          f"({before - len(df):,} dropped for insufficient history)")
    print(f"[features] horizon = {horizon} day(s) ahead")
    print(f"[features] positive rate at t+{horizon}: {df[TARGET].mean():.2%}")

    if save:
        out = ROOT / "data/processed" / f"features_h{horizon}.parquet"
        df.to_parquet(out, index=False)
        print(f"[features] wrote {out}")

    return df, feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1,
                    help="predict flood this many days ahead")
    args = ap.parse_args()
    build(horizon=args.horizon)
