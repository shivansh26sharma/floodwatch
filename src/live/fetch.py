"""
Fetch CURRENT and FORECAST conditions for the study districts.

    python -m src.live.fetch            # writes data/live/conditions.csv

WHAT THIS GETS
--------------
Same providers as the historical download, different endpoints:

  rainfall   Open-Meteo forecast API — recent observed days plus a 16-day
             precipitation forecast
  discharge  Open-Meteo flood API (GloFAS) — recent days plus forecast

We ask for a long run of PAST days as well as the forecast, because the
model's features need history: 30-day rolling windows, a 15-day antecedent
wetness sum, 7-day lags. Predicting "flood onset in the next 7 days" from
today requires roughly two months of lead-in, so that is what we fetch.

THE HONEST CAVEAT — TRAIN/SERVE SKEW
------------------------------------
The model was trained on ERA5 reanalysis and GloFAS historical reanalysis.
These endpoints serve a DIFFERENT product: recent analysis and
numerical-weather-prediction forecast. They are produced by related systems
and are broadly comparable, but they are not the same series — resolution,
bias and the treatment of extremes all differ, and forecast skill decays
with lead time.

So live predictions carry more uncertainty than the backtest suggests, and
the gap widens the further ahead you look. State this in the report. The
rigorous fix is to retrain on the forecast product itself, which needs an
archive of past forecasts that we do not have. Saying so plainly is better
than quietly presenting live output as if it were backtested.
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import CFG, ROOT
from src.data import districts as dref
from src.data.download import PAUSE_SECONDS, fetch_json

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"

# Lead-in needed to build the features: 30-day rollings + 15-day antecedent
# + 7-day lags, with headroom.
PAST_DAYS = 92
FORECAST_DAYS = 16


def fetch_live(districts: dict[str, tuple[float, float]],
               past_days: int = PAST_DAYS,
               forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    frames = []

    for name, (lat, lon) in districts.items():
        print(f"  live  {name}")

        rain = fetch_json(FORECAST_URL, {
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum",
            "past_days": past_days, "forecast_days": forecast_days,
            "timezone": "Asia/Kolkata",
        })["daily"]
        rain_df = pd.DataFrame({
            "date": pd.to_datetime(rain["time"]),
            "district": name,
            "rainfall_mm": rain["precipitation_sum"],
        })
        time.sleep(PAUSE_SECONDS)

        flood = fetch_json(FLOOD_URL, {
            "latitude": lat, "longitude": lon,
            "daily": "river_discharge",
            "past_days": past_days, "forecast_days": min(forecast_days, 30),
        })["daily"]
        flow_df = pd.DataFrame({
            "date": pd.to_datetime(flood["time"]),
            "district": name,
            "river_discharge_m3s": flood["river_discharge"],
        })
        time.sleep(PAUSE_SECONDS)

        frames.append(rain_df.merge(flow_df, on=["date", "district"], how="outer"))

    out = pd.concat(frames, ignore_index=True).sort_values(["district", "date"])

    # Mark which rows are forecast rather than observed — the dashboard and
    # the report both need to distinguish them.
    today = pd.Timestamp(datetime.now(timezone.utc).astimezone().date())
    out["is_forecast"] = out["date"] > today
    return out.reset_index(drop=True)


def run(past_days: int = PAST_DAYS, forecast_days: int = FORECAST_DAYS) -> Path:
    names = dref.selected()
    coords = dref.coords(names)
    print(f"[live] fetching {len(coords)} districts: "
          f"{past_days}d history + {forecast_days}d forecast")

    df = fetch_live(coords, past_days, forecast_days)

    out_dir = ROOT / "data/live"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "conditions.csv"
    df.to_csv(path, index=False)

    observed = int((~df["is_forecast"]).sum())
    forecast = int(df["is_forecast"].sum())
    print(f"[live] {len(df):,} rows  ({observed:,} observed, {forecast:,} forecast)")
    print(f"[live] range {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"[live] wrote {path}")
    print("[live] NOTE: this is forecast/analysis data, not the ERA5 "
          "reanalysis the model trained on — see the module docstring.")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--past-days", type=int, default=PAST_DAYS)
    ap.add_argument("--forecast-days", type=int, default=FORECAST_DAYS)
    args = ap.parse_args()
    run(past_days=args.past_days, forecast_days=args.forecast_days)
