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
    """Expand Dartmouth Flood Observatory events into district-days.

    DFO publishes one row per FLOOD EVENT with a begin date, an end date and
    a coarse affected area. We need one row per (district, day) with a 0/1
    flag, so each event is expanded across its date range and matched to
    districts.

    MATCHING IS THE WEAK POINT AND YOUR REPORT MUST SAY SO.
    DFO's spatial descriptions are free text at country/state/region
    granularity, not district polygons. We therefore match on a text search
    of the event's location fields against district names, and fall back to
    marking every district in the region when an event names the state but
    no district. That over-attributes: a flood that hit only coastal
    Alappuzha may get labelled across the state.

    This is the single largest source of label noise in the project.
    Quantify it if you can (how many events matched a specific district
    versus the state-wide fallback — this function reports exactly that) and
    discuss it as a limitation.

    Column names in DFO exports have changed over the years, so the finder
    below is deliberately tolerant. If it cannot find what it needs it
    raises with the actual column list so you can map them by hand.
    """
    path = ROOT / CFG.data.real.labels
    df = _require(path, "DFO flood archive")

    def find(*candidates: str) -> str | None:
        lowered = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            for low, original in lowered.items():
                if cand in low:
                    return original
        return None

    begin_col = find("began", "begin", "start")
    end_col = find("ended", "end date", "finish")
    country_col = find("country")
    place_col = find("location", "area", "detailed", "place", "admin")

    if not (begin_col and end_col):
        raise ValueError(
            f"Could not find begin/end date columns in {path.name}.\n"
            f"Columns present: {list(df.columns)}\n"
            f"Map them by hand in _load_real_labels()."
        )

    from src.data import districts as dref
    districts = dref.selected()
    region = str(CFG.project.region).lower()
    # Also match on state names, since DFO usually names the state, not the
    # district. A district is hit when its own name OR its state appears.
    states = {d: dref.INDIA_FLOOD_DISTRICTS[d][2].lower() for d in districts}

    work = df.copy()
    work["_begin"] = pd.to_datetime(work[begin_col], errors="coerce")
    work["_end"] = pd.to_datetime(work[end_col], errors="coerce")
    work = work.dropna(subset=["_begin", "_end"])

    # Keep events in the right country, when the column exists.
    if country_col:
        work = work[
            work[country_col].astype(str).str.lower().str.contains("india", na=False)
        ]

    text_cols = [c for c in (place_col, country_col) if c]
    if text_cols:
        work["_text"] = (
            work[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
        )
    else:
        work["_text"] = ""

    # ---- event -> districts ------------------------------------------
    # Three tiers, most specific first:
    #   1. the district is named outright          -> that district
    #   2. only its state is named                 -> every district we
    #                                                 study in that state
    #   3. only "India" is named                   -> SKIPPED. With a
    #      nationwide study set, attributing a vague national event to all
    #      70 districts would manufacture tens of thousands of false
    #      positives. Dropping it loses a few events; keeping it would
    #      poison the labels.
    def mentions(row_text: str) -> tuple[list[str], str]:
        hits = [d for d in districts if d.lower() in row_text]
        if hits:
            return hits, "district"
        by_state = [d for d, st in states.items() if st in row_text]
        if by_state:
            return by_state, "state"
        return [], "none"

    specific, statewide, skipped, rows = 0, 0, 0, []
    for _, ev in work.iterrows():
        matched, how = mentions(ev["_text"])
        if not matched:
            skipped += 1
            continue
        if how == "district":
            specific += 1
        else:
            statewide += 1

        for day in pd.date_range(ev["_begin"], ev["_end"], freq="D"):
            for district in matched:
                rows.append({"date": day, "district": district,
                             "flood_occurred": 1})

    if not rows:
        raise ValueError(
            f"No DFO events matched region {CFG.project.region!r} or its "
            f"districts.\nChecked {len(work)} dated India events.\n"
            f"Inspect the location column ({place_col!r}) and widen the "
            f"matching in _load_real_labels()."
        )

    flood_days = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["date", "district"])
    )

    # Every other district-day in range is a zero, not a missing value.
    start = min(flood_days["date"].min(), pd.Timestamp(CFG.project.start_date))
    end = max(flood_days["date"].max(), pd.Timestamp(CFG.project.end_date))
    grid = pd.MultiIndex.from_product(
        [pd.date_range(start, end, freq="D"), districts],
        names=["date", "district"],
    ).to_frame(index=False)

    out = grid.merge(flood_days, on=["date", "district"], how="left")
    out["flood_occurred"] = out["flood_occurred"].fillna(0).astype(int)

    print(f"[labels] matched {specific} events by district name, "
          f"{statewide} by state name; skipped {skipped} too vague to place")
    print(f"[labels] {int(out['flood_occurred'].sum()):,} flood district-days "
          f"({out['flood_occurred'].mean():.2%} of the grid)")
    if statewide:
        print(f"[labels] NOTE: {statewide} events were spread across every "
              f"studied district in the named state, because DFO gave no "
              f"district. That over-attributes — report it as label noise.")
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
