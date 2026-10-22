"""
Score CURRENT conditions and produce a forward-looking flood risk.

    python -m src.live.predict                 # score today
    python -m src.live.predict --days-ahead 5  # also project 5 days out

WHAT THE NUMBER MEANS
---------------------
The model answers one question: **does a flood begin in this district
within the next 7 days?** Scoring it with features built from the most
recent observation gives today's answer. Scoring it with features built
from a day inside the forecast window gives that day's answer — so a
district can show rising risk several days before anything happens.

HOW THE FEATURES ARE BUILT
--------------------------
Exactly as in training, using the SAVED reference (climatology, percentile
grids, medians). Rebuilding those from the live window would silently
rescale the inputs; see src/features/reference.py.

TWO THINGS TO BE HONEST ABOUT
-----------------------------
1. Train/serve skew. The model learned on ERA5/GloFAS reanalysis; this
   scores forecast and recent-analysis data from related but different
   products. Live numbers are less reliable than the backtest implies.

2. Forecast decay. A feature built from day t+5 depends on five days of
   forecast rainfall. Precipitation forecast skill falls off quickly, so
   risk for distant days is softer than risk for today. The output marks
   how much of each row's history was forecast rather than observed, so
   that uncertainty is visible instead of hidden.

Nothing here is an evacuation order. It is decision support, and the
operating threshold was chosen for a 20:1 miss-to-false-alarm cost, which
means deliberately many false alarms.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG, ROOT
from src.features.reference import FeatureReference


def _load_model(name: str, horizon: int):
    path = ROOT / CFG.outputs.models / f"risk_{name}_h{horizon}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Train one first:\n"
            f"    python -m src.models.evaluate --horizon {horizon}"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _alert_threshold(name: str, horizon: int) -> float:
    path = ROOT / CFG.outputs.reports / f"summary_h{horizon}.json"
    if not path.exists():
        return 0.5
    with open(path) as fh:
        summary = json.load(fh)
    for m in summary.get("models", []):
        if m["model"] == name:
            return float(m["threshold"])
    return 0.5


def load_conditions() -> pd.DataFrame:
    path = ROOT / "data/live/conditions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No live conditions at {path}. Fetch them first:\n"
            f"    python -m src.live.fetch"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["district", "date"]).reset_index(drop=True)


def build_live_features(conditions: pd.DataFrame,
                        reference: FeatureReference) -> pd.DataFrame:
    """Rebuild the training features over the live window.

    Reuses the same functions the training pipeline uses, so there is one
    definition of every feature rather than a second one that drifts.
    """
    from src.data import districts as dref
    from src.features import build_features as bf

    df = conditions.copy()

    # Terrain is static; attach it from the district table rather than
    # refetching elevation on every scoring run.
    meta = dref.meta(sorted(df["district"].unique()))
    terrain_path = ROOT / CFG.data.real.terrain
    if terrain_path.exists():
        terrain = pd.read_csv(terrain_path)
        keep = ["district", "mean_elevation_m", "mean_slope_deg",
                "terrain_ruggedness_m", "lat", "lon"]
        df = df.merge(terrain[keep], on="district", how="left")
        df = df.merge(meta, on="district", how="left")
    else:
        raise FileNotFoundError(
            f"{terrain_path} is required for live scoring (elevation, slope "
            f"and ruggedness are model features). Run:\n"
            f"    python -m src.data.download"
        )

    df = bf._lag_and_roll(df)
    df = bf._add_departure(df, reference.climatology)
    df = bf._add_seasonality(df)
    df = bf._add_basin_context(df)
    df = bf._add_relative_intensity(df, train_end=None, reference=reference)
    df = bf._add_region_dummies(df)
    df = bf._add_terrain_interactions(df)

    # Any feature the model expects but this window could not produce (a
    # basin dummy for a basin not present, say) is filled with the training
    # median — the same thing training did for its own gaps.
    for col in reference.features:
        if col not in df.columns:
            df[col] = reference.medians.get(col, 0.0)
    df[reference.features] = df[reference.features].fillna(reference.medians)
    return df


def score(model_name: str = "random_forest", horizon: int = 1,
          days_ahead: int = 0) -> pd.DataFrame:
    conditions = load_conditions()
    reference = FeatureReference.load()
    bundle = _load_model(model_name, horizon)
    model, feats = bundle["model"], bundle["features"]
    threshold = _alert_threshold(model_name, horizon)

    missing = [f for f in feats if f not in reference.features]
    if missing:
        raise ValueError(
            f"The saved model expects features the reference does not know: "
            f"{missing[:5]}. Model and reference are out of step — rebuild "
            f"both:\n    python -m src.features.build_features --horizon {horizon}"
            f"\n    python -m src.models.evaluate --horizon {horizon}"
        )

    live = build_live_features(conditions, reference)

    # Score the most recent observed day, plus any requested forecast days.
    today = conditions.loc[~conditions["is_forecast"], "date"].max()
    targets = [today] + [
        today + pd.Timedelta(days=d) for d in range(1, days_ahead + 1)
    ]
    rows = live[live["date"].isin(targets)].copy()
    if rows.empty:
        raise ValueError("No scoreable rows — is data/live/conditions.csv current?")

    rows["risk"] = model.predict_proba(rows[feats])[:, 1]
    rows["alert"] = rows["risk"] >= threshold
    rows["days_ahead"] = (rows["date"] - today).dt.days

    # How much of each row's 30-day lead-in was forecast rather than observed.
    fc = conditions.set_index(["district", "date"])["is_forecast"]
    def forecast_share(r):
        window = pd.date_range(r["date"] - pd.Timedelta(days=29), r["date"])
        vals = [fc.get((r["district"], d), False) for d in window]
        return float(np.mean(vals))
    rows["forecast_share"] = rows.apply(forecast_share, axis=1)

    out = rows[["district", "state", "basin", "date", "days_ahead", "risk",
                "alert", "forecast_share", "rainfall_mm",
                "river_discharge_m3s"]].sort_values(
        ["days_ahead", "risk"], ascending=[True, False]).reset_index(drop=True)

    out_dir = ROOT / "data/live"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "risk.csv", index=False)

    # ---- console report ------------------------------------------------
    print("\n" + "=" * 78)
    print(f"LIVE FLOOD RISK — probability a flood BEGINS within 7 days")
    print(f"model: {model_name} · alert threshold {threshold:.2f} · "
          f"scored {today.date()}")
    print("=" * 78)
    for d in sorted(out["days_ahead"].unique()):
        day = out[out["days_ahead"] == d]
        label = "today" if d == 0 else f"+{d}d ({day['date'].iloc[0].date()})"
        fshare = day["forecast_share"].mean()
        print(f"\n{label}   [{fshare:.0%} of the input window is forecast]")
        top = day.head(8)
        for _, r in top.iterrows():
            flag = "ALERT" if r["alert"] else "     "
            print(f"  {flag} {r['district']:16s} {r['risk']:6.1%}  "
                  f"rain {r['rainfall_mm']:6.1f}mm  "
                  f"Q {r['river_discharge_m3s']:8.1f} m3/s")
        n_alert = int(day["alert"].sum())
        print(f"  ... {n_alert}/{len(day)} districts at alert")

    print(f"\nwrote {out_dir / 'risk.csv'}")
    print("Decision support, not an evacuation order. The threshold is set "
          "for a 20:1 miss:false-alarm cost, so false alarms are expected.")
    print("=" * 78 + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="random_forest")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--days-ahead", type=int, default=0,
                    help="also score this many forecast days beyond today")
    args = ap.parse_args()
    score(model_name=args.model, horizon=args.horizon, days_ahead=args.days_ahead)
