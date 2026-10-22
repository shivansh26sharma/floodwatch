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

# A feed older than this is refetched. Half a day: rainfall that matters
# arrives faster than that, and the providers update more often than we
# need, so there is no point hammering a free API every page view.
MAX_AGE_HOURS = 12


def conditions_path() -> Path:
    return ROOT / "data/live/conditions.csv"


def fetched_at() -> datetime | None:
    """When the feed on disk was written, or None if there isn't one."""
    path = conditions_path()
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def age_hours() -> float | None:
    stamp = fetched_at()
    if stamp is None:
        return None
    return (datetime.now(timezone.utc) - stamp).total_seconds() / 3600.0


def _one_district(name: str, lat: float, lon: float,
                  past_days: int, forecast_days: int) -> pd.DataFrame:
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

    return rain_df.merge(flow_df, on=["date", "district"], how="outer")


def fetch_live(districts: dict[str, tuple[float, float]],
               past_days: int = PAST_DAYS,
               forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    """Pull every district, tolerating a few failures but not a dead network.

    FAIL FAST ON A BLOCKED NETWORK. `fetch_json` retries with exponential
    backoff, which is right for a flaky connection and badly wrong for a
    blocked one: 22 districts x 2 endpoints x a full backoff ladder is
    something like twenty minutes of a frozen page to establish a fact the
    first request already established. So if the first district yields
    nothing at all, give up immediately and let the caller fall back to the
    cached feed. Once ANY district has come back, individual failures are
    tolerated — a partial feed scores the districts it has.
    """
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for name, (lat, lon) in districts.items():
        print(f"  live  {name}")
        try:
            frames.append(_one_district(name, lat, lon, past_days, forecast_days))
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            if not frames:
                raise RuntimeError(
                    f"Live feed unreachable — the first district ({name}) "
                    f"returned nothing: {exc}"
                ) from exc
            print(f"    skipped {name}: {exc}")

    if failed:
        print(f"[live] WARNING {len(failed)} district(s) missing from this "
              f"feed: {', '.join(failed)}")

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

    path = conditions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    observed = int((~df["is_forecast"]).sum())
    forecast = int(df["is_forecast"].sum())
    print(f"[live] {len(df):,} rows  ({observed:,} observed, {forecast:,} forecast)")
    print(f"[live] range {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"[live] wrote {path}")
    print("[live] NOTE: this is forecast/analysis data, not the ERA5 "
          "reanalysis the model trained on — see the module docstring.")
    return path


def ensure_conditions(max_age_hours: float = MAX_AGE_HOURS,
                      force: bool = False) -> tuple[Path, bool]:
    """Make sure a usable feed is on disk. Returns (path, refetched).

    The dashboard calls this on start-up. `data/live/` is gitignored, so a
    fresh deploy has no feed at all and has to go and get one; after that
    the file is reused until it ages out. Nothing here raises on a network
    failure if a usable feed already exists — a stale answer clearly
    labelled as stale beats a dead page.
    """
    path = conditions_path()
    age = age_hours()
    if not force and age is not None and age < max_age_hours:
        return path, False

    try:
        run()
        return path, True
    except Exception:
        if path.exists():
            return path, False
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--past-days", type=int, default=PAST_DAYS)
    ap.add_argument("--forecast-days", type=int, default=FORECAST_DAYS)
    args = ap.parse_args()
    run(past_days=args.past_days, forecast_days=args.forecast_days)
