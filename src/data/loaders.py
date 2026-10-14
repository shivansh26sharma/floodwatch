"""
Data loaders.

Each loader returns a tidy DataFrame with a canonical schema, regardless of
whether the source is the synthetic fixture or the real download. That way the
rest of the pipeline never knows or cares which it is — flipping
`data.source` in config.yaml is the only change needed.

CANONICAL SCHEMAS
-----------------
rainfall : date (datetime64), district (str), rainfall_mm (float)
river    : date (datetime64), district (str), river_level_m (float)
terrain  : district (str), mean_elevation_m, mean_slope_deg,
           drainage_density, lat, lon
labels   : date (datetime64), district (str), flood_occurred (int 0/1)

WHEN REAL DATA ARRIVES
----------------------
Fill in the `_load_real_*` functions below. They are the ONLY place that
needs to know the quirks of each source's file format. Everything
downstream is already written and tested.
"""

from pathlib import Path

import pandas as pd

from src.config import CFG, ROOT

CANONICAL = {
    "rainfall": ["date", "district", "rainfall_mm"],
    "river": ["date", "district", "river_level_m"],
    "terrain": ["district", "mean_elevation_m", "mean_slope_deg",
                "drainage_density", "lat", "lon"],
    "labels": ["date", "district", "flood_occurred"],
}


# ----------------------------------------------------------------------
# Fixture loaders
# ----------------------------------------------------------------------

def _load_fixture(name: str) -> pd.DataFrame:
    path = ROOT / CFG.data.fixture.path / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture {path} missing. Generate it first:\n"
            f"    python -m src.data.make_fixture"
        )
    # comment="#" skips the NOT-REAL-DATA banner line
    df = pd.read_csv(path, comment="#")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ----------------------------------------------------------------------
# Real-data loaders  --  FILL THESE IN (week 1-2)
# ----------------------------------------------------------------------

def _load_real_rainfall() -> pd.DataFrame:
    """IMD daily rainfall, via data.gov.in.

    TODO(week 1): IMD files are usually wide (one column per day) or
    district-aggregated monthly. Reshape to long format here with
    `pd.melt`, then rename to the canonical schema. Keep the raw file
    untouched in data/raw/ — all cleaning happens in this function so it
    is reviewable and reproducible.
    """
    path = ROOT / CFG.data.real.rainfall
    df = pd.read_csv(path)
    raise NotImplementedError(
        f"Implement _load_real_rainfall(). Loaded {path.name} with columns:\n"
        f"  {list(df.columns)}\n"
        f"Reshape to: {CANONICAL['rainfall']}"
    )


def _load_real_river() -> pd.DataFrame:
    """India-WRIS / CWC gauge readings.

    TODO(week 1): WRIS exports one file per gauge station. You will need
    a station -> district mapping (nearest-district join on lat/lon using
    the terrain table), then aggregate to district-day with max() —
    the peak reading is what matters for flooding, not the mean.
    """
    path = ROOT / CFG.data.real.river
    df = pd.read_csv(path)
    raise NotImplementedError(
        f"Implement _load_real_river(). Loaded {path.name} with columns:\n"
        f"  {list(df.columns)}\n"
        f"Reshape to: {CANONICAL['river']}"
    )


def _load_real_terrain() -> pd.DataFrame:
    """Terrain statistics derived from the SRTM DEM.

    TODO(week 2): This is a one-off preprocessing job, not a loader.
    Use rasterio + district boundary shapefiles to compute zonal
    statistics (mean elevation, mean slope) per district, and
    pysheds/WhiteboxTools for drainage density. Write the result once to
    data/raw/srtm_terrain.csv and this loader just reads it.
    """
    path = ROOT / CFG.data.real.terrain
    return pd.read_csv(path)


def _load_real_labels() -> pd.DataFrame:
    """Flood event labels from the Dartmouth Flood Observatory archive.

    TODO(week 1): DFO gives one row per EVENT with a begin and end date
    and an affected area. You must expand each event into district-days:
    for every district intersecting the event area, mark every day from
    begin to end as flood_occurred = 1. All other district-days are 0.

    Be careful: DFO's spatial extents are coarse. Document how you
    assigned events to districts — this is the single biggest source of
    label noise in the project and your report should say so.
    """
    path = ROOT / CFG.data.real.labels
    df = pd.read_csv(path)
    raise NotImplementedError(
        f"Implement _load_real_labels(). Loaded {path.name} with columns:\n"
        f"  {list(df.columns)}\n"
        f"Expand events to district-days: {CANONICAL['labels']}"
    )


_REAL = {
    "rainfall": _load_real_rainfall,
    "river": _load_real_river,
    "terrain": _load_real_terrain,
    "labels": _load_real_labels,
}


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def load(name: str) -> pd.DataFrame:
    """Load one canonical table, from fixture or real source per config."""
    if name not in CANONICAL:
        raise KeyError(f"Unknown table {name!r}. Expected one of {list(CANONICAL)}")

    source = CFG.data.source
    if source == "fixture":
        df = _load_fixture(name)
    elif source == "real":
        df = _REAL[name]()
    else:
        raise ValueError(f"config data.source must be 'fixture' or 'real', got {source!r}")

    missing = set(CANONICAL[name]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Table {name!r} from source {source!r} is missing columns: {sorted(missing)}"
        )
    return df[CANONICAL[name]]


def load_all() -> dict[str, pd.DataFrame]:
    return {name: load(name) for name in CANONICAL}


def is_fixture() -> bool:
    """True when running on synthetic data. Used to stamp warnings on outputs."""
    return CFG.data.source == "fixture"
