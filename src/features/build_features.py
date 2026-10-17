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
    # basin/state are kept as columns for grouping and reporting; basin
    # enters the model only through its one-hot dummies, and raw district
    # identity never does.
    "basin", "state", "_onset",
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
    df["river_discharge_delta_1d"] = g["river_discharge_m3s"].diff(1)
    df["river_discharge_delta_3d"] = g["river_discharge_m3s"].diff(3)

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


def _add_relative_intensity(df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """Where does today sit in THIS district's own historical distribution?

    The single biggest weakness of the raw features: discharge is measured
    in m3/s and varies by two orders of magnitude between the Brahmaputra
    and a Konkan stream. 800 m3/s is a drought in Dibrugarh and a disaster
    in Sangli. A model fed raw magnitudes has to learn that mapping through
    the terrain features, and it does it badly.

    So we add percentile ranks computed against each district's own record.
    The reference distribution is the TRAINING PERIOD ONLY — ranking against
    the full record would let the test years inform their own percentile and
    leak the future into the present.
    """
    df = df.sort_values(ID_COLS).copy()
    ref = df[df["date"] <= train_end]

    for col in ("rainfall_mm", "river_discharge_m3s", "antecedent_rain_mm"):
        if col not in df.columns:
            continue
        out = np.full(len(df), np.nan)
        for district, idx in df.groupby("district").groups.items():
            sample = np.sort(ref.loc[ref["district"] == district, col].dropna().to_numpy())
            if sample.size < 50:
                continue
            values = df.loc[idx, col].to_numpy(dtype=float)
            # searchsorted gives the rank of each value within the training
            # distribution; divide by n to get a percentile in [0, 1].
            ranks = np.searchsorted(sample, values, side="right") / sample.size
            out[df.index.get_indexer(idx)] = ranks
        df[f"{col}_pctile"] = out

    return df


def _add_basin_context(df: pd.DataFrame) -> pd.DataFrame:
    """What is happening elsewhere in the same river basin today?

    Floods propagate. Rain falling on upper Assam today is the Brahmaputra's
    problem downstream tomorrow, and a district-only view cannot see that
    coming. Basin-wide aggregates are a cheap proxy for upstream state
    without building an actual routing model.

    Computed per (basin, date) across districts, then each district's own
    contribution is left in — these are context, not leakage: every value
    comes from the SAME day's observations, which are available at
    prediction time, and the target is strictly in the future.
    """
    if "basin" not in df.columns:
        return df

    for col in ("rainfall_mm", "river_discharge_m3s"):
        if col not in df.columns:
            continue
        grp = df.groupby(["basin", "date"])[col]
        df[f"basin_{col}_mean"] = grp.transform("mean")
        df[f"basin_{col}_max"] = grp.transform("max")
        # How extreme is this district relative to its basin today?
        df[f"{col}_vs_basin"] = df[col] / (df[f"basin_{col}_mean"] + 1e-6)

    if "antecedent_rain_mm" in df.columns:
        df["basin_antecedent_mean"] = (
            df.groupby(["basin", "date"])["antecedent_rain_mm"].transform("mean")
        )
    return df


def _add_region_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot the river basin.

    The Brahmaputra floods every monsoon; the Ghaggar almost never does.
    Without a regime marker the model has to infer that from terrain alone,
    which it does badly. Basin is coarse enough (12 values) not to let the
    model memorise individual districts — that would be district-identity
    leakage dressed up as a feature, and it would collapse the moment you
    pointed the system at a district it had never seen.
    """
    if "basin" not in df.columns:
        return df
    dummies = pd.get_dummies(df["basin"], prefix="basin").astype(int)
    return pd.concat([df, dummies], axis=1)


def _add_terrain_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Physically motivated: the same rainfall is more dangerous on flat,
    # low-lying, well-drained-into terrain.
    df["rain_x_inv_slope"] = df["rainfall_mm"] / (df["mean_slope_deg"] + 1.0)
    df["antecedent_x_ruggedness"] = df["antecedent_rain_mm"] * df["terrain_ruggedness_m"]
    df["elevation_inv"] = 1.0 / (df["mean_elevation_m"] + 10.0)
    return df


def _add_target(df: pd.DataFrame, horizon: int,
                mode: str = "onset_window", window: int = 7) -> pd.DataFrame:
    """Build the prediction target.

    TWO MODES, AND THE CHOICE MATTERS MORE THAN ANY MODEL HYPERPARAMETER.

    "ongoing"      flood_occurred at t + horizon. The obvious target, and
                   the one that quietly broke our evaluation. DFO's records
                   changed convention over time: events in 2000-2014 average
                   20.0 days long, events in 2017-2023 average 8.6 days.
                   Same number of events per year, same polygon sizes —
                   only the duration bookkeeping changed. Training on
                   "ongoing" therefore learns 20-day floods and is tested on
                   8.6-day floods, which is why the positive rate collapsed
                   from 7.16% (train) to 1.45% (test). That is a labelling
                   artefact masquerading as a hard problem.

    "onset_window" does a flood BEGIN in the next `window` days? A flood's
                   start date is a fact about the world; how long DFO chose
                   to keep the event open is bookkeeping. This target is
                   therefore far more stable across the record, and it is
                   also the operationally useful question — you want warning
                   BEFORE the water arrives, not confirmation that it is
                   still there on day twelve.

    Onset is defined per district as a 1 whose previous day was 0, so a
    continuing flood does not keep re-triggering.
    """
    df = df.sort_values(ID_COLS).copy()
    g = df.groupby("district", group_keys=False)

    if mode == "ongoing":
        df[TARGET] = g["flood_occurred"].shift(-horizon)
        return df

    if mode != "onset_window":
        raise ValueError(f"Unknown target mode {mode!r}")

    prev = g["flood_occurred"].shift(1).fillna(0)
    df["_onset"] = ((df["flood_occurred"] == 1) & (prev == 0)).astype(int)

    # Does an onset occur in [t+horizon, t+horizon+window-1]?
    # Reverse the series, take a trailing rolling max, reverse back — that
    # is a forward-looking window without a Python loop.
    def forward_any(s: pd.Series) -> pd.Series:
        return (
            s[::-1]
            .shift(horizon)
            .rolling(window, min_periods=1)
            .max()[::-1]
        )

    df[TARGET] = df.groupby("district", group_keys=False)["_onset"].apply(forward_any)
    return df.drop(columns=["_onset"])


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

def build(horizon: int = 1, save: bool = True,
          target_mode: str | None = None,
          target_window: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    from src.data.loaders import effective_source, precomputed_file
    path = (precomputed_file() if effective_source() == "precomputed"
            else ROOT / CFG.data.processed)
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
    df = _add_basin_context(df)
    df = _add_relative_intensity(df, pd.Timestamp(CFG.split.train_end))
    df = _add_region_dummies(df)
    df = _add_terrain_interactions(df)
    target_mode = target_mode or getattr(CFG.features, "target_mode", "onset_window")
    target_window = target_window or int(getattr(CFG.features, "target_window", 7))
    df = _add_target(df, horizon, mode=target_mode, window=target_window)

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
    if target_mode == "onset_window":
        print(f"[features] target = flood ONSET within {target_window}d, "
              f"starting t+{horizon}")
    else:
        print(f"[features] target = flood ongoing at t+{horizon}")
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
    ap.add_argument("--target-mode", default=None,
                    choices=["onset_window", "ongoing"])
    ap.add_argument("--target-window", type=int, default=None)
    args = ap.parse_args()
    build(horizon=args.horizon, target_mode=args.target_mode,
          target_window=args.target_window)
