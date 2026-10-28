"""
FloodWatch — operational console.

Run:
    streamlit run dashboard/app.py

FOUR SURFACES, ONE PIPELINE
---------------------------
  Live operations     today and the next six days, scored from the current
                      rainfall and river-discharge feed
  Historical replay   any day in 2000-2023 replayed against the recorded
                      flood archive — this is where the model is *validated*
  Model performance   the backtest: lead time, duty cycle, calibration,
                      cost curve. The numbers that decide whether the live
                      tab deserves to be believed.
  Data & method       sources, licences, target definition, leakage controls

The split matters. The live tab is the part that looks impressive; the
model tab is the part that makes it defensible. A console that shows the
first without the second is a demo, not a system.

WHAT THE RISK NUMBER MEANS
--------------------------
P(a flood BEGINS in this district within the next 7 days). Not "is it
flooding" — onset. The alert threshold is not 0.5: it is chosen on the
validation split to minimise expected cost at a 20:1 miss-to-false-alarm
ratio, which deliberately buys false alarms to avoid misses.
"""

import inspect
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CFG  # noqa: E402
from src.data import loaders as _loaders  # noqa: E402
from src.pipeline import ensure_artifacts  # noqa: E402

st.set_page_config(page_title="FloodWatch — flood risk console",
                   page_icon="🌊", layout="wide")

HORIZON = 1
# Logistic regression, not the forest. The forest is quieter (10.5% duty
# cycle against 14.7%) but it warns half the events at a third of the
# notice, and at its operating threshold it peaked at 0.017 during the
# Kerala flood of August 2018 — the largest event in the record. A default
# that misses that is the wrong default.
DEFAULT_MODEL = "logistic"
LIVE_DAYS = 6          # today plus six forecast days
FEED_MAX_AGE_HOURS = 12


# ----------------------------------------------------------------------
# Streamlit has renamed its width argument. Pass whichever one this
# version understands rather than pinning everyone to one release.
# ----------------------------------------------------------------------

def _width_kw(fn) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


DF_KW = _width_kw(st.dataframe)
CHART_KW = _width_kw(st.altair_chart)


def chart(spec) -> None:
    st.altair_chart(spec, **CHART_KW)


def table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, hide_index=True, **DF_KW, **kwargs)


def risk_map(frame: pd.DataFrame, threshold: float, height: int = 430,
             scale_to_threshold: bool = True) -> None:
    """District risk map, framed on the districts themselves.

    The view is set with `fit_bounds`, not with a zoom level guessed from
    the latitude span. The guess was wrong the first time this ran on the
    deploy — the map opened on the whole world — because a zoom level has
    to be chosen before Leaflet knows how wide its container is, and the
    hosted layout is not the local one. Bounds do not have that problem:
    they are applied after layout, so the frame is correct by construction
    whatever the container turns out to be.
    """
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Install folium + streamlit-folium for the map view.")
        table(frame[["district", "risk", "lat", "lon"]])
        return

    pts = frame.dropna(subset=["lat", "lon"])
    if pts.empty:
        st.info("No district coordinates available for this day.")
        return

    fmap = folium.Map(location=[pts["lat"].mean(), pts["lon"].mean()],
                      zoom_start=5, tiles="OpenStreetMap", control_scale=True)

    for _, r in pts.iterrows():
        colour = ("#b91c1c" if r["risk"] >= threshold
                  else "#f59e0b" if r["risk"] >= threshold * 0.6
                  else "#15803d")
        # At a 1.6% base rate, sizing markers against 1.0 makes every
        # district the same dot. Size against the alert threshold instead.
        radius = (7 + 18 * min(1.0, float(r["risk"]) / max(threshold, 1e-6))
                  if scale_to_threshold else 8 + 16 * float(r["risk"]))
        tip = (f"<b>{r['district']}</b><br>"
               f"Risk: {r['risk']:.1%}<br>"
               f"Rain: {r['rainfall_mm']:.1f} mm<br>"
               f"Discharge: {r['river_discharge_m3s']:,.0f} m³/s")
        if "forecast_share" in pts.columns:
            tip += f"<br>Forecast share: {r['forecast_share']:.0%}"
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=radius, color=colour,
            fill=True, fill_opacity=0.6, weight=1, tooltip=tip,
        ).add_to(fmap)

    pad = 0.75
    fmap.fit_bounds([[pts["lat"].min() - pad, pts["lon"].min() - pad],
                     [pts["lat"].max() + pad, pts["lon"].max() + pad]])

    # returned_objects=[] stops the component sending map state back on
    # every pan, which otherwise reruns the whole script mid-drag.
    st_folium(fmap, height=height, use_container_width=True,
              returned_objects=[])


# ----------------------------------------------------------------------
# Chrome
#
# All decoration is inline SVG data URIs and CSS gradients — no external
# image, no CDN, nothing to fail on a slow or offline deploy. The water
# motif is kept to the page edges and the sidebar: the moment a texture
# runs under a number, the number gets harder to read, and this is a page
# people are meant to read numbers off.
# ----------------------------------------------------------------------

# Two offset wave profiles. Encoded rather than raw (%3C, %23) so the URI
# survives any CSS parser.
def _wave(c1: str, c2: str, o1: str = ".55", o2: str = ".40") -> str:
    return (
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
        "viewBox='0 0 1200 160' preserveAspectRatio='none'%3E"
        f"%3Cpath d='M0,78 C150,128 330,28 600,78 C870,128 1050,28 1200,78 "
        f"L1200,160 L0,160 Z' fill='%23{c1}' opacity='{o1}'/%3E"
        f"%3Cpath d='M0,104 C210,54 390,154 600,104 C810,54 990,154 1200,104 "
        f"L1200,160 L0,160 Z' fill='%23{c2}' opacity='{o2}'/%3E"
        "%3C/svg%3E"
    )


PAGE_WAVE = _wave("bfdbfe", "7dd3fc", ".32", ".22")
BAR_WAVE = _wave("2563eb", "0ea5e9", ".16", ".12")
SIDE_WAVE = _wave("38bdf8", "60a5fa", ".22", ".16")

st.markdown(
    f"""
    <style>
      /* ---- page ---------------------------------------------------- */
      [data-testid="stAppViewContainer"] {{
        background:
          radial-gradient(1100px 460px at 8% -12%, #dbeafe 0%, rgba(219,234,254,0) 62%),
          radial-gradient(900px 380px at 92% -6%, #e0f2fe 0%, rgba(224,242,254,0) 58%),
          linear-gradient(180deg, #f8fafc 0%, #eef5fb 100%);
      }}
      /* A shallow tide along the bottom of the viewport. Fixed, behind
         everything, and click-through, so it decorates without
         interfering. */
      [data-testid="stAppViewContainer"]::after {{
        content:""; position:fixed; left:0; right:0; bottom:0; height:150px;
        background-image:url("{PAGE_WAVE}");
        background-size:1200px 150px; background-repeat:repeat-x;
        background-position:0 bottom;
        animation:fw-drift 26s linear infinite;
        pointer-events:none; z-index:0;
      }}
      @keyframes fw-drift {{
        from {{background-position:0 bottom;}}
        to   {{background-position:-1200px bottom;}}
      }}
      .block-container {{padding-top:1.1rem; max-width:1500px;
        position:relative; z-index:1;}}
      [data-testid="stHeader"] {{background:transparent;}}

      /* ---- status bar ---------------------------------------------- */
      .fw-bar {{position:relative; overflow:hidden;
        display:flex; align-items:center; gap:.6rem; flex-wrap:wrap;
        border:1px solid rgba(37,99,235,.16); border-radius:18px;
        padding:.95rem 1.2rem 1.15rem;
        background:linear-gradient(118deg,#ffffff 0%,#f0f7ff 52%,#e4f1fd 100%);
        box-shadow:0 10px 30px -18px rgba(13,42,92,.45);
        margin-bottom:1.2rem;}}
      .fw-bar::after {{
        content:""; position:absolute; left:0; right:0; bottom:-2px;
        height:46px;
        background-image:url("{BAR_WAVE}");
        background-size:1200px 46px; background-repeat:repeat-x;
        animation:fw-drift 18s linear infinite;
        pointer-events:none;}}
      .fw-bar > * {{position:relative; z-index:1;}}
      .fw-brand {{font-weight:800; font-size:1.3rem; letter-spacing:-.025em;
        color:#0b2545; margin-right:.55rem;}}
      .fw-brand span {{background:linear-gradient(92deg,#2563eb,#0ea5e9);
        -webkit-background-clip:text; background-clip:text;
        -webkit-text-fill-color:transparent;}}
      .fw-sub {{color:#5b7a9b; font-size:.82rem; margin-right:.5rem;}}
      .fw-grow {{flex:1 1 auto;}}

      /* ---- pills ---------------------------------------------------- */
      .fw-pill {{font-size:.71rem; font-weight:700; letter-spacing:.05em;
        text-transform:uppercase; padding:.3rem .68rem; border-radius:999px;
        border:1px solid rgba(148,163,184,.55); color:#334155;
        background:rgba(255,255,255,.88);
        box-shadow:0 1px 2px rgba(15,23,42,.05);
        backdrop-filter:blur(4px); white-space:nowrap;}}
      .fw-pill.ok   {{border-color:#86efac; color:#15803d;
        background:linear-gradient(180deg,#f0fdf4,#dcfce7);}}
      .fw-pill.warn {{border-color:#fcd34d; color:#92400e;
        background:linear-gradient(180deg,#fffbeb,#fef3c7);}}
      .fw-pill.bad  {{border-color:#fca5a5; color:#b91c1c;
        background:linear-gradient(180deg,#fef2f2,#fee2e2);}}
      .fw-dot {{height:7px; width:7px; border-radius:50%; display:inline-block;
        margin-right:.4rem; vertical-align:middle;
        box-shadow:0 0 0 3px rgba(22,163,74,.14);}}
      .fw-pill.ok .fw-dot {{animation:fw-pulse 2.4s ease-in-out infinite;}}
      @keyframes fw-pulse {{
        0%,100% {{box-shadow:0 0 0 2px rgba(22,163,74,.18);}}
        50%     {{box-shadow:0 0 0 6px rgba(22,163,74,0);}}
      }}

      /* ---- sidebar: the dark console rail --------------------------- */
      [data-testid="stSidebar"] {{
        background:linear-gradient(185deg,#0b2545 0%,#12315c 55%,#163e75 100%);
        border-right:1px solid rgba(255,255,255,.08);}}
      [data-testid="stSidebar"]::after {{
        content:""; position:absolute; left:0; right:0; bottom:0; height:150px;
        background-image:url("{SIDE_WAVE}");
        background-size:900px 150px; background-repeat:repeat-x;
        animation:fw-drift 32s linear infinite;
        pointer-events:none;}}
      [data-testid="stSidebar"] .stMarkdown,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color:#dbeafe !important;}}
      [data-testid="stSidebar"] hr {{border-color:rgba(255,255,255,.16);}}
      [data-testid="stSidebar"] h3 {{letter-spacing:.02em;}}

      /* ---- buttons --------------------------------------------------- */
      .stButton > button, [data-testid="stBaseButton-secondary"] {{
        background:linear-gradient(135deg,#0ea5e9 0%,#2563eb 100%);
        color:#ffffff !important; border:0; border-radius:12px;
        padding:.62rem 1.05rem; font-weight:700; letter-spacing:.01em;
        box-shadow:0 8px 18px -8px rgba(37,99,235,.85),
                   inset 0 1px 0 rgba(255,255,255,.22);
        transition:transform .16s ease, box-shadow .16s ease, filter .16s ease;}}
      .stButton > button:hover {{transform:translateY(-2px); filter:brightness(1.06);
        box-shadow:0 14px 26px -10px rgba(37,99,235,.95),
                   inset 0 1px 0 rgba(255,255,255,.3);}}
      .stButton > button:active {{transform:translateY(0);
        box-shadow:0 5px 12px -8px rgba(37,99,235,.9);}}
      .stButton > button p {{color:#ffffff !important; font-weight:700;}}

      /* ---- tabs ------------------------------------------------------ */
      [data-testid="stTabs"] [role="tablist"] {{gap:.35rem;
        align-items:center; border-bottom:0 !important;
        background:rgba(255,255,255,.65); padding:.32rem; border-radius:14px;
        border:1px solid rgba(148,163,184,.3); backdrop-filter:blur(6px);
        box-shadow:0 6px 18px -16px rgba(13,42,92,.8);}}
      [data-testid="stTab"] {{font-weight:700; border-radius:10px;
        padding:.42rem .9rem; margin:0 !important; color:#475569;
        border-bottom:0 !important; transition:all .15s ease;}}
      [data-testid="stTab"] p {{font-weight:700; color:#475569;}}
      [data-testid="stTab"]:hover {{background:rgba(37,99,235,.08);}}
      [data-testid="stTab"]:hover p {{color:#1d4ed8;}}
      [data-testid="stTab"][aria-selected="true"] {{
        background:linear-gradient(135deg,#0ea5e9,#2563eb);
        box-shadow:0 6px 14px -8px rgba(37,99,235,.9);}}
      [data-testid="stTab"][aria-selected="true"] p {{color:#ffffff !important;}}
      [data-testid="stTabs"] [role="tablist"] + div,
      [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        background:transparent !important;}}

      /* ---- horizon picker (segmented control) ------------------------ */
      button[data-variant="segmented_control"] {{border-radius:10px !important;
        font-weight:650; transition:all .15s ease;}}
      button[data-variant="segmented_control"]:hover {{
        border-color:#93c5fd !important;}}
      button[data-variant="segmented_control"][aria-checked="true"] {{
        background:linear-gradient(135deg,#0ea5e9,#2563eb) !important;
        border-color:transparent !important;
        box-shadow:0 6px 14px -8px rgba(37,99,235,.9);}}
      button[data-variant="segmented_control"][aria-checked="true"] p,
      button[data-variant="segmented_control"][data-selected="true"] p {{
        color:#ffffff !important; font-weight:700;}}

      /* ---- metric cards ---------------------------------------------- */
      div[data-testid="stMetric"] {{position:relative; overflow:hidden;
        border:1px solid rgba(148,163,184,.3); border-radius:16px;
        padding:.95rem 1.05rem .85rem;
        background:linear-gradient(170deg,#ffffff 0%,#f6fbff 100%);
        box-shadow:0 10px 24px -20px rgba(13,42,92,.75);
        transition:transform .16s ease, box-shadow .16s ease;}}
      div[data-testid="stMetric"]::before {{content:""; position:absolute;
        left:0; top:0; height:3px; width:100%;
        background:linear-gradient(90deg,#0ea5e9,#2563eb);}}
      div[data-testid="stMetric"]:hover {{transform:translateY(-3px);
        box-shadow:0 18px 34px -22px rgba(13,42,92,.9);}}
      [data-testid="stMetricLabel"] p {{font-size:.76rem; color:#64748b;
        text-transform:uppercase; letter-spacing:.06em; font-weight:700;}}
      div[data-testid="stMetricValue"] {{color:#0b2545;}}

      /* ---- panels ---------------------------------------------------- */
      [data-testid="stAlert"] {{border-radius:14px;
        box-shadow:0 8px 20px -18px rgba(13,42,92,.8);}}
      [data-testid="stExpander"] details {{border-radius:14px;
        border:1px solid rgba(148,163,184,.35);
        background:rgba(255,255,255,.75);}}
      [data-testid="stDataFrame"] {{border-radius:12px; overflow:hidden;
        border:1px solid rgba(148,163,184,.3);}}
      iframe {{border-radius:14px;}}

      .fw-note {{color:#64748b; font-size:.8rem; line-height:1.45;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def pill(text: str, kind: str = "", dot: str = "") -> str:
    mark = f'<span class="fw-dot" style="background:{dot}"></span>' if dot else ""
    return f'<span class="fw-pill {kind}">{mark}{text}</span>'


# ----------------------------------------------------------------------
# Bootstrap + loading
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="First run on this server — building the pipeline…")
def _bootstrap(horizon: int):
    """On a fresh cloud deploy the build artifacts don't exist yet.

    Build them once from the committed dataset. Cached as a resource so
    this runs once per server process, not once per page view.
    """
    return ensure_artifacts(horizon=horizon)


_bootstrap(HORIZON)


@st.cache_data
def load_features(horizon: int) -> pd.DataFrame:
    path = ROOT / "data/processed" / f"features_h{horizon}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource
def load_model(name: str, horizon: int):
    path = ROOT / CFG.outputs.models / f"risk_{name}_h{horizon}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


@st.cache_data
def scored_frame(name: str, horizon: int) -> pd.DataFrame:
    df = load_features(horizon)
    bundle = load_model(name, horizon)
    if df.empty or bundle is None:
        return pd.DataFrame()
    df = df.copy()
    df["risk"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    return df


@st.cache_data
def summary(horizon: int) -> dict:
    path = ROOT / CFG.outputs.reports / f"summary_h{horizon}.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


@st.cache_data
def report_csv(stem: str) -> pd.DataFrame:
    path = ROOT / CFG.outputs.reports / f"{stem}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def available_models(horizon: int) -> list[str]:
    """Only models that are in THIS run's summary.

    Report CSVs from an earlier run can survive on disk — an xgboost file
    left behind after xgboost stopped importing, say. Driving the menu off
    the summary means the console never offers a model it cannot score.
    """
    models = [m["model"] for m in summary(horizon).get("models", [])]
    return models or [DEFAULT_MODEL]


def metrics_for(name: str, horizon: int) -> dict:
    for m in summary(horizon).get("models", []):
        if m["model"] == name:
            return m
    return {}


def alert_threshold(name: str, horizon: int) -> float:
    return float(metrics_for(name, horizon).get("threshold", 0.5))


# ----------------------------------------------------------------------
# Live feed
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Fetching current rainfall and river levels…")
def feed_bootstrap(max_age_hours: int) -> dict:
    """Make sure there is a live feed on disk, once per server process.

    `data/live/` is gitignored, so a fresh deploy starts with nothing and
    has to go and get a feed. After that the file is reused until it ages
    out, because a free public API does not deserve a request per pageview.
    """
    from src.live import fetch as lf
    try:
        path, refetched = lf.ensure_conditions(max_age_hours=max_age_hours)
        return {"ok": True, "refetched": refetched, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "refetched": False, "error": str(exc)}


@st.cache_data(show_spinner="Scoring current conditions…")
def live_frames(model: str, horizon: int, days: int, _stamp: float):
    """Score the live window. `_stamp` is the feed mtime — it exists purely
    so that refetching the feed invalidates this cache."""
    from src.live import predict as lp
    scored = lp.score(model_name=model, horizon=horizon,
                      days_ahead=days, verbose=False)
    return scored, lp.load_conditions()


def feed_age_hours() -> float | None:
    from src.live import fetch as lf
    return lf.age_hours()


def feed_stamp() -> float:
    from src.live import fetch as lf
    path = lf.conditions_path()
    return path.stat().st_mtime if path.exists() else 0.0


def human_age(hours: float | None) -> str:
    if hours is None:
        return "no feed"
    if hours < 1:
        return f"{int(hours * 60)} min ago"
    if hours < 48:
        return f"{hours:.0f} h ago"
    return f"{hours / 24:.0f} d ago"


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------

model_choices = available_models(HORIZON)
with st.sidebar:
    st.markdown("### Console")
    MODEL_NAME = st.selectbox(
        "Risk model", model_choices,
        index=model_choices.index(DEFAULT_MODEL)
        if DEFAULT_MODEL in model_choices else 0,
        format_func=lambda s: s.replace("_", " ").title(),
    )
    st.caption(
        "Logistic regression is the default: it warns 65% of recorded events "
        "a median 13 days ahead, against 51% at 5 days for the forest. "
        "Neither model's output is a calibrated probability — the Model tab "
        "says what that means."
    )
    st.divider()
    if st.button("↻ Refresh live feed", **_width_kw(st.button)):
        from src.live import fetch as lf
        with st.spinner("Pulling 22 districts from Open-Meteo — ~2 minutes…"):
            try:
                lf.ensure_conditions(force=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Fetch failed: {exc}")
        st.rerun()
    st.caption(f"Feed updated {human_age(feed_age_hours())}. Auto-refetched "
               f"when older than {FEED_MAX_AGE_HOURS} h.")

data = scored_frame(MODEL_NAME, HORIZON)

if data.empty:
    st.error(
        "No scored data found. Run the pipeline first:\n\n"
        "```\n"
        "python -m src.data.build_dataset\n"
        "python -m src.features.build_features --horizon 1\n"
        "python -m src.models.evaluate --horizon 1\n"
        "```"
    )
    st.stop()

REQUIRED = ["district", "date", "risk", "rainfall_mm", "river_discharge_m3s",
            "lat", "lon"]
missing_cols = [c for c in REQUIRED if c not in data.columns]
if missing_cols:
    st.error(
        f"The cached data is from an older version of the pipeline — it is "
        f"missing: `{'`, `'.join(missing_cols)}`.\n\n"
        f"Delete `data/processed/` and `outputs/models/` and reload, or "
        f"redeploy so the bootstrap rebuilds them."
    )
    st.caption(f"Columns present: {', '.join(sorted(data.columns))}")
    st.stop()

threshold = alert_threshold(MODEL_NAME, HORIZON)
metrics = metrics_for(MODEL_NAME, HORIZON)

# ----------------------------------------------------------------------
# Status bar
# ----------------------------------------------------------------------

source = _loaders.effective_source()
n_districts = data["district"].nunique()
n_basins = data["basin"].nunique() if "basin" in data.columns else 0
span = f"{data['date'].min():%Y}–{data['date'].max():%Y}"

if source == "precomputed":
    src_pill = pill("real data", "ok", "#16a34a")
elif source == "real":
    src_pill = pill("real data", "ok", "#16a34a")
else:
    src_pill = pill("synthetic fixture — numbers are fake", "bad", "#dc2626")

age = feed_age_hours()
if age is None:
    feed_pill = pill("live feed offline", "bad", "#dc2626")
elif age <= FEED_MAX_AGE_HOURS:
    feed_pill = pill(f"feed {human_age(age)}", "ok", "#16a34a")
else:
    feed_pill = pill(f"feed stale · {human_age(age)}", "warn", "#f59e0b")

st.markdown(
    '<div class="fw-bar">'
    '<span class="fw-brand">🌊 Flood<span>Watch</span></span>'
    f'<span class="fw-sub">{CFG.project.region} · {n_districts} districts · '
    f'{n_basins} river basins · {span}</span>'
    '<span class="fw-grow"></span>'
    f'{src_pill}{feed_pill}'
    f'{pill(MODEL_NAME.replace("_", " "))}'
    f'{pill(f"alert ≥ {threshold:.2f}")}'
    '</div>',
    unsafe_allow_html=True,
)

if source == "fixture":
    reason = _loaders.fallback_reason()
    st.error(
        (f"**Real data was requested but is not available here.** {reason}\n\n"
         if reason else "**Running on the synthetic test fixture.** ")
        + "Every number below is synthetic and exists only to prove the "
          "pipeline runs end to end.",
        icon="🚨",
    )

tab_live, tab_replay, tab_model, tab_data = st.tabs([
    "◉  Live operations",
    "⏮  Historical replay",
    "◆  Model performance",
    "▤  Data & method",
])


# ======================================================================
# TAB 1 — LIVE OPERATIONS
# ======================================================================

with tab_live:
    feed = feed_bootstrap(FEED_MAX_AGE_HOURS)

    if not feed["ok"]:
        st.error(
            f"**Could not reach the live feed and there is no cached copy.**\n\n"
            f"`{feed['error']}`\n\nThe other three tabs run entirely on "
            f"committed data and are unaffected.",
            icon="📡",
        )
    else:
        try:
            live, conditions = live_frames(MODEL_NAME, HORIZON, LIVE_DAYS,
                                           feed_stamp())
        except Exception as exc:  # noqa: BLE001
            live, conditions = pd.DataFrame(), pd.DataFrame()
            st.error(f"**Live scoring failed.** `{exc}`", icon="⚠️")

        if not live.empty:
            import altair as alt

            today = live.loc[live["days_ahead"] == 0, "date"].iloc[0]
            now = live[live["days_ahead"] == 0]
            horizon_max = live.groupby("district", as_index=False)["risk"].max()
            # The discharge forecast can be shorter than the rainfall
            # forecast, so the furthest scoreable day is whatever actually
            # came back — not whatever we asked for.
            max_day = int(live["days_ahead"].max())
            far_share = float(
                live.loc[live["days_ahead"] == max_day, "forecast_share"].mean())

            # ---- alert line ------------------------------------------
            alerting = live[live["alert"]]
            if alerting.empty:
                st.success(
                    f"**No district at alert through {max_day} days.** "
                    f"Highest projected risk is "
                    f"{horizon_max['risk'].max():.1%} in "
                    f"{horizon_max.loc[horizon_max['risk'].idxmax(), 'district']}, "
                    f"against an alert threshold of {threshold:.0%}.",
                    icon="✅",
                )
            else:
                first = (alerting.sort_values("days_ahead")
                         .groupby("district", as_index=False).first())
                lines = ", ".join(
                    f"**{r.district}** "
                    + ("today" if r.days_ahead == 0 else f"+{r.days_ahead}d")
                    for r in first.itertuples()
                )
                st.error(f"**{len(first)} district(s) at alert:** {lines}",
                         icon="🚨")

            # ---- KPI row ---------------------------------------------
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("At alert today", int(now["alert"].sum()),
                      f"{int(alerting['district'].nunique())} over {max_day}d",
                      delta_color="off")
            top_now = now.sort_values("risk", ascending=False).iloc[0]
            k2.metric("Highest risk today", f"{top_now['risk']:.1%}",
                      top_now["district"], delta_color="off")
            peak = horizon_max.loc[horizon_max["risk"].idxmax()]
            k3.metric("Peak over horizon", f"{peak['risk']:.1%}",
                      peak["district"], delta_color="off")
            k4.metric(f"Forecast share at +{max_day}d", f"{far_share:.0%}",
                      "of the input window", delta_color="off")
            st.markdown(
                '<p class="fw-note">Risk is the probability a flood '
                '<b>begins</b> in that district within seven days. '
                '“Forecast share” is how much of the 30-day feature window '
                'is numerical forecast rather than observed — the honest '
                'measure of how soft a distant day’s number is.</p>',
                unsafe_allow_html=True,
            )

            st.divider()

            # ---- horizon picker --------------------------------------
            days = sorted(live["days_ahead"].unique().tolist())

            def day_label(d: int) -> str:
                stamp = today + pd.Timedelta(days=int(d))
                return "Today" if d == 0 else f"+{d}d · {stamp:%d %b}"

            if hasattr(st, "segmented_control"):
                picked = st.segmented_control(
                    "Horizon", days, format_func=day_label, default=days[0])
                picked = days[0] if picked is None else picked
            else:
                picked = st.radio("Horizon", days, format_func=day_label,
                                  horizontal=True)

            day = live[live["days_ahead"] == picked].sort_values(
                "risk", ascending=False)

            left, right = st.columns([3, 2])

            with left:
                st.markdown(f"**Risk map — {day_label(picked)}**")
                risk_map(day, threshold, height=430)

            with right:
                st.markdown("**Watchlist**")
                view = day[["district", "basin", "risk", "rainfall_mm",
                            "river_discharge_m3s", "forecast_share"]].copy()
                # column_config formats the raw value, so a 0–1 share has to
                # become 0–100 before it can be shown as a percentage.
                view["forecast_share"] = view["forecast_share"] * 100
                view["risk"] = view["risk"] * 100
                view = view.rename(
                    columns={"district": "District", "basin": "Basin",
                             "risk": "Risk", "rainfall_mm": "Rain (mm)",
                             "river_discharge_m3s": "Q (m³/s)",
                             "forecast_share": "Forecast"})
                cap = max(float(day["risk"].max()), threshold) * 100
                table(view, column_config={
                    "Risk": st.column_config.ProgressColumn(
                        "Risk", min_value=0.0, max_value=cap, format="%.2f%%"),
                    "Forecast": st.column_config.NumberColumn(
                        "Forecast", format="%.0f%%", help=(
                            "Share of the 30-day feature window that is "
                            "forecast rather than observed")),
                })
                st.caption(
                    f"Bars are scaled to the alert threshold ({threshold:.0%}), "
                    f"not to 100% — at a 1.6% base rate a 0–1 axis would show "
                    f"nothing."
                )

            st.divider()

            # ---- trajectory ------------------------------------------
            st.markdown("**Risk trajectory — where the next week is heading**")
            watch = horizon_max.nlargest(6, "risk")["district"].tolist()
            traj = live[live["district"].isin(watch)].copy()
            traj["pct"] = traj["risk"]
            lines = alt.Chart(traj).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("days_ahead:O", title="days ahead"),
                y=alt.Y("pct:Q", title="P(onset within 7 days)",
                        axis=alt.Axis(format="%")),
                color=alt.Color("district:N", title="District"),
                tooltip=[alt.Tooltip("district:N", title="District"),
                         alt.Tooltip("date:T", title="Date"),
                         alt.Tooltip("pct:Q", title="Risk", format=".2%"),
                         alt.Tooltip("forecast_share:Q", title="Forecast share",
                                     format=".0%")],
            ).properties(height=300)
            rule = alt.Chart(
                pd.DataFrame({"t": [threshold]})
            ).mark_rule(color="#b91c1c", strokeDash=[5, 4]).encode(y="t:Q")
            chart(lines + rule)
            st.caption(
                "Six districts with the highest projected risk. The dashed "
                "line is the alert threshold. A rising line is the whole "
                "point of the system — it is a warning before anything has "
                "happened."
            )

            st.divider()

            # ---- district detail -------------------------------------
            st.markdown("**District detail — observed history and forecast**")
            sel = st.selectbox("District", sorted(live["district"].unique()),
                               index=sorted(live["district"].unique()).index(
                                   str(peak["district"])))
            hist = conditions[
                (conditions["district"] == sel)
                & (conditions["date"] >= today - pd.Timedelta(days=45))
            ].copy()
            hist["Rainfall"] = hist["rainfall_mm"]

            bars = alt.Chart(hist).mark_bar(size=6).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("Rainfall:Q", title="Rainfall (mm/day)"),
                color=alt.condition(alt.datum.is_forecast,
                                    alt.value("#93c5fd"), alt.value("#2563eb")),
                tooltip=[alt.Tooltip("date:T", title="Date"),
                         alt.Tooltip("Rainfall:Q", title="Rain (mm)",
                                     format=".1f"),
                         alt.Tooltip("is_forecast:N", title="Forecast")],
            ).properties(height=160)
            flow = alt.Chart(hist).mark_line(color="#b45309",
                                             strokeWidth=2).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("river_discharge_m3s:Q", title="Discharge (m³/s)"),
                tooltip=[alt.Tooltip("date:T", title="Date"),
                         alt.Tooltip("river_discharge_m3s:Q",
                                     title="Q (m³/s)", format=",.0f")],
            ).properties(height=160)
            today_rule = alt.Chart(
                pd.DataFrame({"d": [today]})
            ).mark_rule(color="#0f172a", strokeDash=[3, 3]).encode(x="d:T")

            chart(bars + today_rule)
            chart(flow + today_rule)
            st.caption(
                "Dark bars are observed rainfall, light bars are forecast; the "
                "dashed vertical line is the last observed day. Discharge is "
                "GloFAS, a modelled routing of that rainfall through the river "
                "network — not a gauge reading."
            )

            with st.expander("What makes a live number weaker than a backtested one"):
                st.markdown(
                    f"""
**Train/serve skew.** The model was fitted on ERA5 reanalysis and GloFAS
historical reanalysis — products built *after the fact*, with observations
assimilated. This tab feeds it recent analysis and NWP forecast from the
same provider family. Related series, not identical ones: resolution,
extreme-value behaviour and bias all differ. The rigorous fix is to retrain
on an archive of past *forecasts*, which we do not have, so the correct
move is to say so rather than present live output as if it were backtested.

**Forecast decay.** A feature built for +{max_day}d draws {far_share:.0%}
of its 30-day window from forecast rainfall. Precipitation skill falls off
fast, so the far end of the trajectory chart is much softer than the near
end. That is why `forecast_share` is a column and not a footnote.

**Not an evacuation order.** The threshold is tuned for a
{CFG.evaluation.cost_false_negative}:1 miss-to-false-alarm cost. It is
*supposed* to raise alarms that come to nothing. Use it to decide where to
look, never as the sole basis for acting.
                    """
                )


# ======================================================================
# TAB 2 — HISTORICAL REPLAY
# ======================================================================

with tab_replay:
    st.markdown("**Replay any day in the record against the flood archive.**")
    st.markdown(
        '<p class="fw-note">This is the tab that does the validating. Every '
        'day here has a known outcome in the Dartmouth Flood Observatory '
        'archive, so the risk shown can be checked rather than believed.</p>',
        unsafe_allow_html=True,
    )

    dates = sorted(data["date"].unique())
    # Open on the peak-risk day rather than the middle of the series. A demo
    # that loads on a dry Tuesday in 2015 with zero alerts wastes the first
    # ten seconds of every viva.
    peak_date = data.loc[data["risk"].idxmax(), "date"]
    selected = st.select_slider(
        "Date", options=dates, value=peak_date,
        format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"),
    )
    st.caption("Opens on the highest-risk day in the record.")

    rday = data[data["date"] == selected].copy()
    rday["alert"] = rday["risk"] >= threshold
    rday = rday.sort_values("risk", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts at alert", int(rday["alert"].sum()))
    c2.metric("Highest risk", f"{rday['risk'].max():.1%}",
              rday.iloc[0]["district"])
    c3.metric("Mean rainfall", f"{rday['rainfall_mm'].mean():.1f} mm")
    c4.metric("Mean discharge", f"{rday['river_discharge_m3s'].mean():,.0f} m³/s")

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Risk map**")
        # Replay scores span the full 0-1 range on a big event day, so
        # markers there are sized against 1.0 rather than the threshold.
        risk_map(rday, threshold, height=420, scale_to_threshold=False)

    with right:
        st.markdown("**Ranked districts**")
        ranked = rday[["district", "risk", "rainfall_mm",
                       "river_discharge_m3s", "alert"]].copy()
        ranked["risk"] = ranked["risk"] * 100
        table(
            ranked.rename(columns={
                "district": "District", "risk": "Risk",
                "rainfall_mm": "Rain (mm)",
                "river_discharge_m3s": "Discharge (m³/s)",
                "alert": "Alert"}),
            column_config={
                "Risk": st.column_config.ProgressColumn(
                    "Risk", min_value=0.0, max_value=100.0, format="%.1f%%"),
            },
        )

    st.divider()
    st.markdown("**Trend**")
    district = st.selectbox("District", sorted(data["district"].unique()),
                            key="replay_district")
    window = data[
        (data["district"] == district)
        & (data["date"] >= pd.Timestamp(selected) - pd.Timedelta(days=45))
        & (data["date"] <= pd.Timestamp(selected) + pd.Timedelta(days=15))
    ].set_index("date")
    t1, t2 = st.columns(2)
    t1.line_chart(window[["risk"]], height=220)
    t2.line_chart(window[["rainfall_mm", "river_discharge_m3s"]], height=220)

    st.divider()
    st.markdown("**Why this risk level?**")
    try:
        from src.models.explain import explain_one

        expl = explain_one(district, pd.Timestamp(selected).strftime("%Y-%m-%d"),
                           MODEL_NAME, HORIZON, top_n=6)
        expl["effect"] = expl["contribution"].apply(
            lambda v: "↑ increases risk" if v > 0 else "↓ decreases risk")
        table(expl[["feature", "value", "contribution", "effect"]])
        st.caption(
            "SHAP contributions for this district-day. A risk score an "
            "officer cannot interrogate will not be trusted — and should not be."
        )
    except Exception as exc:  # noqa: BLE001
        st.info(f"Explanation unavailable: {exc}")

    st.divider()
    st.markdown("**Citizen SOS queue**")
    st.info(
        "**Planned module.** Whisper transcribes voice reports, the NLP layer "
        "classifies incident type and severity, and clustered reports appear "
        "here ranked P1–P3 — feeding back into the fusion layer as "
        "corroborating evidence for the districts above.",
        icon="🎙️",
    )


# ======================================================================
# TAB 3 — MODEL PERFORMANCE
# ======================================================================

with tab_model:
    import altair as alt

    if not metrics:
        st.warning("No evaluation summary found. Run "
                   "`python -m src.models.evaluate --horizon 1`.")
    else:
        st.markdown(
            f"**Backtest — {MODEL_NAME.replace('_', ' ').title()}, "
            f"held-out test split.** Threshold fixed on validation, never "
            f"on test."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Events warned",
                  f"{metrics['events_warned']}/{metrics['events_total']}",
                  f"{metrics['event_detection_rate']:.0%} detection")
        m2.metric("Median lead time",
                  f"{metrics['median_lead_time_hours'] / 24:.0f} days",
                  f"mean {metrics['mean_lead_time_hours'] / 24:.1f} d",
                  delta_color="off")
        m3.metric("Alarm duty cycle", f"{metrics['alarm_duty_cycle']:.1%}",
                  "of district-days", delta_color="off")
        m4.metric("PR-AUC", f"{metrics['pr_auc']:.3f}",
                  f"ROC-AUC {metrics['roc_auc']:.3f}", delta_color="off")

        st.warning(
            f"**Why accuracy is not on this page.** Predicting “no flood” "
            f"every single day scores "
            f"{metrics['accuracy_all_negative']:.1%} accuracy on this test "
            f"set, because floods begin on about "
            f"{1 - metrics['accuracy_all_negative']:.1%} of district-days. "
            f"This model scores {metrics['accuracy_at_f1']:.1%} — *lower* "
            f"than the do-nothing baseline — and is nonetheless useful, "
            f"because it warns "
            f"{metrics['event_detection_rate']:.0%} of real events a median "
            f"{metrics['median_lead_time_hours'] / 24:.0f} days early. "
            f"Lead time and duty cycle are the metrics that matter here; "
            f"accuracy is a trap at this base rate.",
            icon="🎯",
        )

        conf = metrics.get("confusion")
        if isinstance(conf, str):
            try:
                conf = json.loads(conf.replace("'", '"'))
            except Exception:  # noqa: BLE001
                conf = None
        if isinstance(conf, dict):
            st.markdown("**Confusion matrix at the operating threshold**")
            cm = pd.DataFrame(
                [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]],
                index=["actual: no onset", "actual: onset"],
                columns=["predicted: quiet", "predicted: alert"],
            )
            st.dataframe(cm, **DF_KW)
            st.caption(
                f"{conf['fp']:,} false alarms against {conf['fn']:,} misses. "
                f"That ratio is a *choice*: the threshold minimises expected "
                f"cost at {CFG.evaluation.cost_false_negative}:1, so the "
                f"model is instructed to accept roughly "
                f"{CFG.evaluation.cost_false_negative} false alarms rather "
                f"than miss one onset. Precision of "
                f"{metrics['precision']:.1%} is the price of that, not a bug."
            )

        st.divider()
        g1, g2 = st.columns(2)

        cost = report_csv(f"cost_curve_{MODEL_NAME}_h{HORIZON}")
        with g1:
            st.markdown("**Cost curve — where the threshold comes from**")
            if cost.empty:
                st.info("No cost curve on disk.")
            else:
                line = alt.Chart(cost).mark_line(strokeWidth=2,
                                                 color="#2563eb").encode(
                    x=alt.X("threshold:Q", title="alert threshold"),
                    y=alt.Y("expected_cost:Q", title="expected cost"),
                    tooltip=[alt.Tooltip("threshold:Q", format=".2f"),
                             alt.Tooltip("expected_cost:Q", format=",.0f"),
                             alt.Tooltip("recall:Q", format=".2f"),
                             alt.Tooltip("precision:Q", format=".3f")],
                ).properties(height=260)
                pick = alt.Chart(
                    pd.DataFrame({"t": [threshold]})
                ).mark_rule(color="#b91c1c", strokeDash=[5, 4]).encode(x="t:Q")
                chart(line + pick)
                st.caption(
                    f"Minimum at {threshold:.2f}, chosen on the **validation** "
                    f"split. Picking it on test would be reading the answer "
                    f"before the exam."
                )

        rel = report_csv(f"reliability_{MODEL_NAME}_h{HORIZON}")
        with g2:
            st.markdown("**Calibration — is a 30% warning a 30% event?**")
            if rel.empty:
                st.info("No reliability curve on disk.")
            else:
                pts = alt.Chart(rel).mark_circle(color="#2563eb").encode(
                    x=alt.X("mean_predicted:Q", title="predicted probability",
                            scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("observed_rate:Q", title="observed onset rate",
                            scale=alt.Scale(domain=[0, 1])),
                    size=alt.Size("n:Q", title="district-days",
                                  legend=None),
                    tooltip=[alt.Tooltip("mean_predicted:Q", format=".3f"),
                             alt.Tooltip("observed_rate:Q", format=".3f"),
                             alt.Tooltip("n:Q", format=",.0f")],
                ).properties(height=260)
                diag = alt.Chart(
                    pd.DataFrame({"x": [0, 1], "y": [0, 1]})
                ).mark_line(color="#94a3b8", strokeDash=[4, 4]).encode(
                    x="x:Q", y="y:Q")
                chart(diag + pts)
                st.caption(
                    f"Points below the diagonal mean the model over-states "
                    f"risk. Brier {metrics['brier']:.3f} — against "
                    f"{1 - metrics['accuracy_all_negative']:.3f} for a model "
                    f"that always predicts zero, which this does not beat."
                )
                st.warning(
                    "**These scores are not calibrated probabilities.** On "
                    "the test split the mean predicted risk is far above the "
                    "actual onset rate — about 16x for logistic regression "
                    "and 7x for the forest — so a reading of 0.30 does not "
                    "mean a 30% chance. Treat the output as a ranking score. "
                    "That inflation is also why the alert threshold sits "
                    "where it does rather than near the base rate. The fix "
                    "is isotonic calibration fitted on the validation split; "
                    "it is listed as future work rather than quietly "
                    "half-done, because it moves every number on this page.",
                    icon="📐",
                )

        leads = report_csv(f"lead_times_{MODEL_NAME}_h{HORIZON}")
        if not leads.empty and "lead_time_hours" in leads.columns:
            st.divider()
            st.markdown("**Lead time on the events we caught**")
            warned = leads.dropna(subset=["lead_time_hours"]).copy()
            if not warned.empty:
                warned["lead_days"] = warned["lead_time_hours"] / 24
                hist = alt.Chart(warned).mark_bar(color="#2563eb").encode(
                    x=alt.X("lead_days:Q", bin=alt.Bin(maxbins=20),
                            title="days of warning before onset"),
                    y=alt.Y("count():Q", title="events"),
                    tooltip=[alt.Tooltip("count():Q", title="events")],
                ).properties(height=220)
                chart(hist)
                st.caption(
                    f"{len(warned)} of {len(leads)} archived onsets were "
                    f"warned. The left tail — warnings of under a day — is "
                    f"the part that would not actually help anyone evacuate, "
                    f"and it is visible here rather than averaged away."
                )

        comp = report_csv(f"model_comparison_h{HORIZON}")
        if not comp.empty:
            st.divider()
            st.markdown("**All models**")
            cols = ["model", "threshold", "event_detection_rate",
                    "median_lead_time_hours", "alarm_duty_cycle", "recall",
                    "precision", "f1", "pr_auc", "roc_auc", "brier"]
            cols = [c for c in cols if c in comp.columns]
            view = comp[comp["model"].isin(model_choices)][cols].copy()
            if "median_lead_time_hours" in view.columns:
                view["median_lead_time_hours"] = (
                    view["median_lead_time_hours"] / 24).round(1)
            view = view.rename(columns={
                "model": "Model", "threshold": "Threshold",
                "event_detection_rate": "Events warned",
                "median_lead_time_hours": "Median lead (d)",
                "alarm_duty_cycle": "Duty cycle", "recall": "Recall",
                "precision": "Precision", "f1": "F1", "pr_auc": "PR-AUC",
                "roc_auc": "ROC-AUC", "brier": "Brier"})
            table(view)


# ======================================================================
# TAB 4 — DATA & METHOD
# ======================================================================

with tab_data:
    st.markdown("**Where every number comes from.**")

    sources = pd.DataFrame([
        {"Layer": "Rainfall (history)", "Product": "ERA5 reanalysis",
         "Via": "Open-Meteo archive API", "Licence": "Free, non-commercial, attribution"},
        {"Layer": "Rainfall (live)", "Product": "Analysis + NWP forecast",
         "Via": "Open-Meteo forecast API", "Licence": "Free, non-commercial, attribution"},
        {"Layer": "River discharge", "Product": "GloFAS",
         "Via": "Open-Meteo flood API", "Licence": "Free, non-commercial, attribution"},
        {"Layer": "Terrain", "Product": "Copernicus DEM (elevation, slope, ruggedness)",
         "Via": "Open-Meteo elevation API", "Licence": "Free, attribution"},
        {"Layer": "Flood labels", "Product": "Dartmouth Flood Observatory event polygons",
         "Via": "Zenodo 10.5281/zenodo.19288170", "Licence": "CC0"},
    ])
    table(sources)
    st.caption(
        "No paid API, no licence that forbids a student project, and nothing "
        "scraped. ERA5 and GloFAS are **reanalysis and model output, not "
        "station gauges** — they smooth extremes, which caps how good the "
        "labels and features can be."
    )

    st.divider()
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**The prediction target**")
        st.markdown(
            f"""
`onset_window`: does a flood **begin** in this district within
{CFG.features.target_window} days, starting at t+{HORIZON}?

Not "is it flooding". That distinction decided the project. DFO records
event *duration*, and a target of "flooding is ongoing" inherits that
duration — which drifted from a mean of 20 days in the early archive to
8.6 days in the later one. The model would have been learning a change in
how the archive was compiled. Onset is a property of the flood, not of the
record-keeping, so the drift disappears.

Labels come from a **spatial join**: each district centroid against the DFO
event polygons (`geopandas.sjoin`, `predicate="within"`), not text matching
on district names.
            """
        )

    with d2:
        st.markdown("**How leakage is kept out**")
        st.markdown(
            f"""
- Every rolling window is `closed="left"` — today's value never enters its
  own feature.
- Rolling and lag features are grouped **per district**, so one district's
  history cannot bleed into another's.
- Day-of-year climatology and the percentile grids are fitted on the
  **training split only**, saved to `feature_reference.pkl`, and reloaded at
  serving time. Re-deriving them live would silently rescale the inputs.
- Splits are chronological, never random:
  train ≤ {CFG.split.train_end} · val ≤ {CFG.split.val_end} · test after.
- The alert threshold is chosen on validation and then frozen.
- `tests/test_no_leakage.py` asserts these rather than trusting them.
            """
        )

    st.divider()
    st.markdown("**Pipeline**")
    st.code(
        "python -m src.data.download                      # ERA5 / GloFAS / DEM\n"
        "python -m src.data.build_dataset                 # join + DFO spatial labels\n"
        "python -m src.features.build_features --horizon 1   # 59 features + reference\n"
        "python -m src.models.evaluate --horizon 1        # train, threshold, report\n"
        "python -m src.live.fetch                         # current + forecast feed\n"
        "python -m src.live.predict --days-ahead 6        # forward risk\n"
        "streamlit run dashboard/app.py                   # this console",
        language="bash",
    )

    figures = getattr(CFG.outputs, "figures", "outputs/figures")
    svg = ROOT / figures / "architecture.svg"
    if svg.exists():
        st.markdown("**Architecture**")
        try:
            st.markdown(f'<div style="background:#fff;border:1px solid #e2e8f0;'
                        f'border-radius:12px;padding:1rem;overflow:auto">'
                        f'{svg.read_text()}</div>', unsafe_allow_html=True)
        except Exception:  # noqa: BLE001
            pass

    st.divider()
    st.markdown(
        '<p class="fw-note">FloodWatch is a student project and decision '
        'support, not an operational warning service. India\'s official '
        'flood forecasts come from the Central Water Commission and the '
        'India Meteorological Department.</p>',
        unsafe_allow_html=True,
    )
