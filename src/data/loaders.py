"""
Data loaders.

Each loader returns a tidy DataFrame with a canonical schema, regardless of
whether the source is the synthetic fixture or the real download. That way the
rest of the pipeline never knows or cares which it is — flipping
`data.source` in config.yaml is the only change needed.

CANONICAL SCHEMAS
-----------------
rainfall : date (datetime64), district (str), rainfall_mm (float)
river    : date (datetime64), district (str), river_discharge_m3s (float)
terrain  : district (str), mean_elevation_m, mean_slope_deg,
           terrain_ruggedness_m, lat, lon
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

_warned: dict[str, bool] = {}

CANONICAL = {
    "rainfall": ["date", "district", "rainfall_mm"],
    "river": ["date", "district", "river_discharge_m3s"],
    "terrain": ["district", "mean_elevation_m", "mean_slope_deg",
                "terrain_ruggedness_m", "lat", "lon", "state", "basin"],
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
# Real-data loaders
# ----------------------------------------------------------------------
# These read what `python -m src.data.download` produced. Each one is the
# ONLY place that knows its source's format — everything downstream sees
# the canonical schema and does not care where the numbers came from.
#
# Swapping in IMD / India-WRIS later means rewriting the body of one
# function, not touching the pipeline.
# ----------------------------------------------------------------------


def _require(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n"
            f"Fetch it first:\n"
            f"    python -m src.data.download"
        )
    return pd.read_csv(path)


def _load_real_rainfall() -> pd.DataFrame:
    """ERA5 daily precipitation (mm), already in long format from download.py.

    Reanalysis, not IMD station gauges — see the note in download.py and say
    so in the report. ERA5's ~25 km grid smooths local extremes, so peak
    intensities during an event like Kerala 2018 are under-represented.
    """
    path = ROOT / CFG.data.real.rainfall
    df = _require(path, "ERA5 rainfall")
    df["date"] = pd.to_datetime(df["date"])
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    return df


def _load_real_river() -> pd.DataFrame:
    """GloFAS daily river discharge (m3/s).

    Discharge, not gauge stage. The column keeps its honest name all the way
    through the pipeline so nobody later reports it as a water level.
    """
    path = ROOT / CFG.data.real.river
    df = _require(path, "GloFAS river discharge")
    df["date"] = pd.to_datetime(df["date"])
    df["river_discharge_m3s"] = pd.to_numeric(
        df["river_discharge_m3s"], errors="coerce"
    )
    return df


def _load_real_terrain() -> pd.DataFrame:
    """Elevation-derived terrain attributes, computed in download.py."""
    path = ROOT / CFG.data.real.terrain
    return _require(path, "terrain")


def _load_real_labels() -> pd.DataFrame:
    """Expand DFO flood events into district-days.

    SPATIAL, NOT TEXTUAL.
    ---------------------
    The redesigned DFO release (v0.9.0, Zenodo) dropped the free-text
    "Detailed Locations" column that older versions carried. Its CSV now
    identifies an event only by COUNTRY, which is useless for district
    labelling — every Indian event would reduce to "India".

    So we use the GeoPackage instead, which carries the actual flood
    POLYGON for each event, and label a district by asking whether its
    point falls inside that polygon. That is strictly better than the old
    text matching: no name-collision noise, and no state-level fallback
    that smeared one district's flood across a whole state.

    Download `Global_Flood_Records.gpkg` from the DFO Zenodo record and
    save it as the path configured in data.real.labels. The data is
    released CC0 / CC-BY-4.0, so you may redistribute it with your repo —
    cite it anyway.

    LIMITATIONS TO REPORT
      - DFO polygons are satellite-derived event footprints, generally
        coarser than a district and sometimes a bounding region rather
        than observed inundation. A district inside the footprint is
        recorded as flooded even if only part of it actually was.
      - We test the district's single headquarters point, not its area.
        A district whose HQ sits just outside a footprint reads as dry.
      - DFO records only LARGE events, so the positive class is rare and
        small local floods are absent entirely.
    """
    path = ROOT / CFG.data.real.labels
    if not path.exists():
        raise FileNotFoundError(
            f"DFO flood records not found at {path}.\n"
            f"Download Global_Flood_Records.gpkg from the DFO Zenodo record\n"
            f"  https://doi.org/10.5281/zenodo.19288170\n"
            f"and save it there. The .gpkg is required (the .csv has no\n"
            f"location column beyond country, so it cannot label districts)."
        )

    if path.suffix.lower() not in {".gpkg", ".geojson", ".json", ".shp"}:
        raise ValueError(
            f"{path.name} looks like a plain table. District labelling needs "
            f"the GeoPackage (Global_Flood_Records.gpkg) — the DFO CSV only "
            f"identifies events by country, which cannot place a district."
        )

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for spatial flood labelling.\n"
            "    pip install geopandas"
        ) from exc

    from src.data import districts as dref

    events = gpd.read_file(path)
    if events.crs is None:
        events = events.set_crs("EPSG:4326")
    events = events.to_crs("EPSG:4326")

    def find(*candidates: str) -> str | None:
        """Match column names ignoring spaces, underscores and case.

        DFO has used "Began"/"Ended", "Start Date"/"End Date" and now
        "BeginDate"/"EndDate" across releases, so normalise before
        comparing rather than guessing one spelling.
        """
        def norm(x: str) -> str:
            return x.lower().replace(" ", "").replace("_", "").strip()

        normalised = {norm(c): c for c in events.columns}
        for cand in candidates:
            target = norm(cand)
            for low, original in normalised.items():
                if target in low:
                    return original
        return None

    begin_col = find("begindate", "startdate", "began", "begin", "start")
    end_col = find("enddate", "ended", "end", "finish")
    if not (begin_col and end_col):
        raise ValueError(
            f"Could not find start/end date columns in {path.name}.\n"
            f"Columns present: {list(events.columns)}"
        )

    events = events.copy()
    events["_begin"] = pd.to_datetime(events[begin_col], errors="coerce")
    events["_end"] = pd.to_datetime(events[end_col], errors="coerce")
    events = events.dropna(subset=["_begin", "_end", "geometry"])

    # Clip to the study window so we are not iterating 40 years of events.
    window_start = pd.Timestamp(CFG.project.start_date)
    window_end = pd.Timestamp(CFG.project.end_date)
    events = events[(events["_end"] >= window_start) & (events["_begin"] <= window_end)]

    names = dref.selected()
    coords = dref.coords(names)
    points = gpd.GeoDataFrame(
        {"district": list(coords)},
        geometry=[Point(lon, lat) for lat, lon in coords.values()],
        crs="EPSG:4326",
    )

    # Spatial join: which districts fall inside which event footprints.
    hits = gpd.sjoin(points, events[["geometry", "_begin", "_end"]],
                     how="inner", predicate="within")

    rows = []
    for _, hit in hits.iterrows():
        lo = max(hit["_begin"], window_start)
        hi = min(hit["_end"], window_end)
        for day in pd.date_range(lo, hi, freq="D"):
            rows.append({"date": day, "district": hit["district"],
                         "flood_occurred": 1})

    matched_events = int(hits.index.size)
    if not rows:
        raise ValueError(
            f"No DFO event footprint contained any studied district.\n"
            f"Checked {len(events)} events in {window_start.date()}..{window_end.date()} "
            f"against {len(names)} district points.\n"
            f"Sanity-check that the GeoPackage really has polygon geometry "
            f"and that your district coordinates are lon/lat correct."
        )

    flood_days = pd.DataFrame(rows).drop_duplicates(subset=["date", "district"])

    # Every other district-day in the window is a zero, not a missing value.
    grid = pd.MultiIndex.from_product(
        [pd.date_range(window_start, window_end, freq="D"), names],
        names=["date", "district"],
    ).to_frame(index=False)

    out = grid.merge(flood_days, on=["date", "district"], how="left")
    out["flood_occurred"] = out["flood_occurred"].fillna(0).astype(int)

    per_district = flood_days.groupby("district").size()
    print(f"[labels] {len(events):,} DFO events in window; "
          f"{matched_events:,} district-event intersections")
    print(f"[labels] {int(out['flood_occurred'].sum()):,} flood district-days "
          f"({out['flood_occurred'].mean():.2%} of the grid)")
    print(f"[labels] {per_district.size}/{len(names)} districts have at least "
          f"one flood day; busiest: "
          f"{', '.join(f'{d} ({n})' for d, n in per_district.nlargest(3).items())}")
    if per_district.size < len(names):
        dry = sorted(set(names) - set(per_district.index))
        print(f"[labels] NOTE: no events ever intersected {len(dry)} districts "
              f"({', '.join(dry[:5])}{'...' if len(dry) > 5 else ''}). They "
              f"contribute only negatives — mention this in limitations.")
    return out


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

    source = effective_source()
    if source == "fixture":
        if CFG.data.source == "real" and not _warned.get("fallback"):
            _warned["fallback"] = True
            print(f"[loaders] {fallback_reason()}")
            print("[loaders] FALLING BACK TO THE SYNTHETIC FIXTURE — "
                  "nothing below is real.")
        df = _load_fixture(name)
    else:
        df = _REAL[name]()

    missing = set(CANONICAL[name]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Table {name!r} from source {source!r} is missing columns: {sorted(missing)}"
        )
    return df[CANONICAL[name]]


def load_all() -> dict[str, pd.DataFrame]:
    return {name: load(name) for name in CANONICAL}


# ----------------------------------------------------------------------
# Effective source: what we can ACTUALLY load right now
# ----------------------------------------------------------------------
# config.yaml says what you WANT. This says what is present. They differ
# in one important case: the repo is deployed with data.source: real, but
# data/raw/ is gitignored, so the hosted app has the config and none of
# the files. Crashing there is useless — the demo should still run, just
# visibly on the fixture.
#
# The fallback is loud, never silent: effective_source() reports it,
# is_fixture() returns True, every pipeline stage prints the fixture
# warning, and the dashboard banner says real data was expected and not
# found. A quiet fallback would be far worse than a crash, because the
# numbers would look real.

REAL_TABLES = ("rainfall", "river", "terrain", "labels")


def _missing_real_files() -> list[str]:
    missing = []
    for name in REAL_TABLES:
        configured = getattr(CFG.data.real, name, None)
        if configured and not (ROOT / configured).exists():
            missing.append(str(configured))
    return missing


def precomputed_file():
    """Path to the committed scoped real dataset."""
    rel = getattr(CFG.data, "precomputed_path", "data/processed/demo_dataset.parquet")
    return ROOT / rel


def effective_source() -> str:
    """What we can actually load right now.

      "real"        the raw ERA5/GloFAS/DFO files are present
      "precomputed" the joined district-day table is committed to the repo.
                    This is REAL data — the raw sources were processed once
                    on a machine that had them, and the 2.7MB result was
                    committed so a deploy can serve real numbers without
                    the ~32MB of source files (which stay gitignored).
      "fixture"     synthetic; nothing below it is real
    """
    configured = CFG.data.source

    if configured == "precomputed":
        return "precomputed" if precomputed_file().exists() else "fixture"

    if configured != "real":
        return "fixture"

    return "fixture" if _missing_real_files() else "real"


def fallback_reason() -> str | None:
    """Why we fell back, if we did — for the dashboard banner and logs."""
    if CFG.data.source == "precomputed" and effective_source() == "fixture":
        return (
            "config.yaml requests the precomputed real dataset but "
            f"{precomputed_file().name} is not present"
        )
    if CFG.data.source != "real":
        return None
    missing = _missing_real_files()
    if not missing:
        return None
    return (
        "config.yaml requests real data but these files are not present: "
        + ", ".join(missing)
    )


def is_fixture() -> bool:
    """True when running on synthetic data. Used to stamp warnings on outputs."""
    return effective_source() == "fixture"


def is_precomputed() -> bool:
    """Real data, loaded from the committed joined table rather than raw files."""
    return effective_source() == "precomputed"
