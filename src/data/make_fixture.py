"""
================================================================
SYNTHETIC TEST FIXTURE GENERATOR  --  *** NOT TRAINING DATA ***
================================================================

This module generates fake-but-plausibly-shaped data so that the
whole pipeline can be built, run and unit-tested BEFORE the real
IMD / WRIS / SRTM / DFO downloads land.

It exists so that:
  - the dashboard team can start in week 1 instead of week 6
  - the pipeline is already debugged when real data arrives
  - tests can run on a fresh clone with no credentials

IT MUST NEVER BE USED TO PRODUCE RESULTS YOU REPORT.
Every number in your report comes from real observational data.
Switch config.yaml -> data.source: real as soon as data lands.

The generator deliberately encodes a few real relationships
(monsoon seasonality, rainfall -> river level lag, saturation
effects) so that models trained on it behave sanely. That is a
debugging convenience, NOT evidence of anything.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG, ROOT

BANNER = "SYNTHETIC FIXTURE - NOT REAL DATA - DO NOT REPORT RESULTS FROM THIS"


def _monsoon_intensity(day_of_year: np.ndarray) -> np.ndarray:
    """Two-peaked Indian monsoon shape: SW monsoon Jun-Sep, NE Oct-Nov."""
    sw = np.exp(-0.5 * ((day_of_year - 200) / 35.0) ** 2)
    ne = 0.45 * np.exp(-0.5 * ((day_of_year - 300) / 22.0) ** 2)
    return 0.06 + sw + ne


def generate(seed: int | None = None) -> dict[str, pd.DataFrame]:
    cfg = CFG
    seed = seed if seed is not None else cfg.data.fixture.seed
    rng = np.random.default_rng(seed)

    from src.data import districts as dref
    districts = dref.selected()
    cap = int(getattr(cfg.data.fixture, "max_districts", 0) or 0)
    if cap and len(districts) > cap:
        districts = districts[:cap]
    dates = pd.date_range(cfg.project.start_date, cfg.project.end_date, freq="D")
    doy = dates.dayofyear.to_numpy()

    # ---- terrain: static per district -------------------------------
    terrain = pd.DataFrame(
        {
            "district": districts,
            "mean_elevation_m": rng.uniform(5, 900, len(districts)).round(1),
            "mean_slope_deg": rng.uniform(0.4, 14.0, len(districts)).round(2),
            "terrain_ruggedness_m": rng.uniform(5.0, 180.0, len(districts)).round(1),
            "lat": [dref.INDIA_FLOOD_DISTRICTS[d][0] for d in districts],
            "lon": [dref.INDIA_FLOOD_DISTRICTS[d][1] for d in districts],
            "state": [dref.INDIA_FLOOD_DISTRICTS[d][2] for d in districts],
            "basin": [dref.INDIA_FLOOD_DISTRICTS[d][3] for d in districts],
        }
    )
    # Low, flat districts flood more easily — encode that in a susceptibility term.
    susceptibility = (
        1.6
        - 0.0011 * terrain["mean_elevation_m"]
        - 0.055 * terrain["mean_slope_deg"]
        + 0.0035 * terrain["terrain_ruggedness_m"]
    ).clip(0.15, 2.0)
    susceptibility.index = terrain["district"]

    rain_rows, river_rows, flood_rows = [], [], []

    for district in districts:
        base = _monsoon_intensity(doy)

        # Rainfall: gamma-distributed wet days, monsoon-modulated.
        wet = rng.random(len(dates)) < (0.18 + 0.52 * base / base.max())
        amounts = rng.gamma(shape=1.6, scale=11.0 * base / base.mean(), size=len(dates))
        rainfall = np.where(wet, amounts, 0.0).round(2)

        # Occasional extreme multi-day spells (the events that matter).
        n_spells = rng.integers(8, 16)
        for _ in range(n_spells):
            start = rng.integers(0, len(dates) - 6)
            if base[start] < 0.5:          # spells happen in monsoon
                continue
            length = rng.integers(2, 6)
            rainfall[start : start + length] += rng.uniform(55, 190, length).round(2)

        rain_rows.append(
            pd.DataFrame({"date": dates, "district": district, "rainfall_mm": rainfall})
        )

        # River discharge (m3/s): lagged, damped routing of rainfall plus a
        # baseflow floor. Magnitudes are chosen to sit in the same range as
        # GloFAS output for catchments this size, so the fixture exercises
        # the pipeline at realistic scale.
        level = np.zeros(len(dates))
        store = 0.0
        for i, r in enumerate(rainfall):
            store = 0.86 * store + 0.14 * r           # catchment storage
            level[i] = 25.0 + 11.0 * store + rng.normal(0, 3.0)
        level = np.clip(level, 1.0, None)
        river_rows.append(
            pd.DataFrame(
                {"date": dates, "district": district,
                 "river_discharge_m3s": level.round(2)}
            )
        )

        # ---- flood labels -------------------------------------------
        # Driven by antecedent wetness x intensity x susceptibility,
        # which is roughly the real physical story.
        antecedent = pd.Series(rainfall).rolling(15, min_periods=1).sum().to_numpy()
        intensity3 = pd.Series(rainfall).rolling(3, min_periods=1).sum().to_numpy()
        drive = (
            0.0022 * antecedent
            + 0.011 * intensity3
            + 0.9 * (level - level.mean()) / (level.std() + 1e-9)
        ) * susceptibility[district]
        # Offset tuned so the positive class is rare (~4-6%), which is the
        # realistic, harder case — real DFO-labelled flood days are rare.
        prob = 1.0 / (1.0 + np.exp(-(drive - 6.0)))
        flooded = rng.random(len(dates)) < prob

        # Floods persist for a few days once they start.
        idx = np.flatnonzero(flooded)
        for i in idx:
            flooded[i : min(i + rng.integers(1, 5), len(flooded))] = True

        flood_rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "district": district,
                    "flood_occurred": flooded.astype(int),
                }
            )
        )

    frames = {
        "rainfall": pd.concat(rain_rows, ignore_index=True),
        "river": pd.concat(river_rows, ignore_index=True),
        "terrain": terrain,
        "labels": pd.concat(flood_rows, ignore_index=True),
    }

    # Realistic mess: real data always has gaps. Punch some holes so the
    # pipeline's missing-value handling is actually exercised.
    for key in ("rainfall", "river"):
        df = frames[key]
        col = "rainfall_mm" if key == "rainfall" else "river_discharge_m3s"
        holes = rng.random(len(df)) < 0.012
        df.loc[holes, col] = np.nan

    return frames


def write(out_dir: Path | None = None, seed: int | None = None) -> Path:
    out_dir = out_dir or ROOT / CFG.data.fixture.path
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = generate(seed)
    for name, df in frames.items():
        path = out_dir / f"{name}.csv"
        with open(path, "w") as fh:
            fh.write(f"# {BANNER}\n")
            df.to_csv(fh, index=False)

    label_rate = frames["labels"]["flood_occurred"].mean()
    print(f"[fixture] wrote {len(frames)} files to {out_dir}")
    print(f"[fixture] {len(frames['labels']):,} district-days, "
          f"flood rate {label_rate:.2%}")
    print(f"[fixture] {BANNER}")
    return out_dir


if __name__ == "__main__":
    write()
