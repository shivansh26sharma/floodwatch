"""
FloodWatch dashboard — Streamlit + Folium.

Run:
    streamlit run dashboard/app.py

This is the integration surface. It is deliberately built and committed
in week 1, against the synthetic fixture, so that whoever owns the
dashboard is never blocked waiting for the modelling team. When the real
models land, nothing here changes — it reads the same parquet and pickle
files.

HISTORICAL REPLAY, NOT LIVE
---------------------------
The date slider replays a chosen historical window day by day. There is
no live data connection, by design: the project validates against
recorded events. Live feeds are future work.

PANELS
  1. Risk map        — districts coloured by predicted flood probability
  2. Risk table      — ranked districts with the alert flag
  3. Trend           — rainfall / river level / predicted risk over time
  4. Why?            — SHAP explanation for the selected district-day
  5. SOS queue       — placeholder for the citizen-report module (week 5-6)
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CFG  # noqa: E402
from src.pipeline import ensure_artifacts  # noqa: E402

st.set_page_config(page_title="FloodWatch", page_icon="🌊", layout="wide")

HORIZON = 1
MODEL_NAME = "random_forest"


@st.cache_resource(show_spinner="First run on this server — building the pipeline…")
def _bootstrap(horizon: int):
    """On a fresh cloud deploy the build artifacts don't exist yet.

    Build them once from the committed fixture. Cached as a resource so
    this runs once per server process, not once per page view.
    """
    return ensure_artifacts(horizon=horizon)


_bootstrap(HORIZON)


# ----------------------------------------------------------------------
# Loading (cached so the slider is responsive)
# ----------------------------------------------------------------------

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


def alert_threshold(name: str, horizon: int) -> float:
    """Read the cost-optimal threshold chosen during evaluation."""
    import json
    path = ROOT / CFG.outputs.reports / f"summary_h{horizon}.json"
    if not path.exists():
        return 0.5
    with open(path) as fh:
        summary = json.load(fh)
    for m in summary.get("models", []):
        if m["model"] == name:
            return float(m["threshold"])
    return 0.5


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

st.title("🌊 FloodWatch")
st.caption(
    f"Multimodal flood risk — {CFG.project.anchor_event} · "
    f"{CFG.project.region} · {HORIZON}-day-ahead forecast"
)

data = scored_frame(MODEL_NAME, HORIZON)

if data.empty:
    st.error(
        "No scored data found. Run the pipeline first:\n\n"
        "```\n"
        "python -m src.data.make_fixture\n"
        "python -m src.data.build_dataset\n"
        "python -m src.features.build_features --horizon 1\n"
        "python -m src.models.evaluate --horizon 1\n"
        "```"
    )
    st.stop()

if CFG.data.source == "fixture":
    st.warning(
        "**Running on the synthetic test fixture.** Every number on this page "
        "is fake and exists only to prove the pipeline works. Set "
        "`data.source: real` in `config.yaml` once the IMD/WRIS/DFO downloads "
        "are in place.",
        icon="⚠️",
    )

threshold = alert_threshold(MODEL_NAME, HORIZON)

# ---- sidebar: the replay control -------------------------------------
with st.sidebar:
    st.header("Replay")
    dates = sorted(data["date"].unique())
    # Open on the peak-risk day rather than the middle of the series.
    # A demo that loads on a dry Tuesday in 2015 with zero alerts wastes
    # the first ten seconds of every viva.
    peak_date = data.loc[data["risk"].idxmax(), "date"]
    selected = st.select_slider(
        "Date",
        options=dates,
        value=peak_date,
        format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"),
    )
    st.caption("Opens on the highest-risk day in the record.")
    st.metric("Alert threshold", f"{threshold:.2f}")
    st.caption(
        "Threshold is chosen on the validation split to minimise expected "
        f"cost at a {CFG.evaluation.cost_false_negative}:1 "
        "miss-to-false-alarm ratio — not set to 0.5."
    )
    st.divider()
    st.caption("Model: " + MODEL_NAME.replace("_", " ").title())

today = data[data["date"] == selected].copy()
today["alert"] = today["risk"] >= threshold
today = today.sort_values("risk", ascending=False)

# ---- headline numbers -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Districts at alert", int(today["alert"].sum()))
c2.metric("Highest risk", f"{today['risk'].max():.1%}",
          today.iloc[0]["district"])
c3.metric("Mean rainfall", f"{today['rainfall_mm'].mean():.1f} mm")
c4.metric("Mean river level", f"{today['river_level_m'].mean():.2f} m")

st.divider()

left, right = st.columns([3, 2])

# ---- 1. map -----------------------------------------------------------
with left:
    st.subheader("Risk map")
    try:
        import folium
        from streamlit_folium import st_folium

        centre = [today["lat"].mean(), today["lon"].mean()]
        fmap = folium.Map(location=centre, zoom_start=8, tiles="cartodbpositron")

        for _, r in today.iterrows():
            colour = (
                "#b91c1c" if r["risk"] >= threshold
                else "#f59e0b" if r["risk"] >= threshold * 0.6
                else "#15803d"
            )
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=8 + 16 * float(r["risk"]),
                color=colour,
                fill=True,
                fill_opacity=0.65,
                weight=1,
                tooltip=(
                    f"<b>{r['district']}</b><br>"
                    f"Risk: {r['risk']:.1%}<br>"
                    f"Rainfall: {r['rainfall_mm']:.1f} mm<br>"
                    f"River: {r['river_level_m']:.2f} m"
                ),
            ).add_to(fmap)

        st_folium(fmap, height=420, use_container_width=True)
    except ImportError:
        st.info("Install folium + streamlit-folium for the map view.")
        st.dataframe(today[["district", "risk", "lat", "lon"]], hide_index=True)

# ---- 2. ranked table --------------------------------------------------
with right:
    st.subheader("Ranked districts")
    st.dataframe(
        today[["district", "risk", "rainfall_mm", "river_level_m", "alert"]]
        .rename(columns={
            "district": "District", "risk": "Risk",
            "rainfall_mm": "Rain (mm)", "river_level_m": "River (m)",
            "alert": "Alert",
        }),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Risk": st.column_config.ProgressColumn(
                "Risk", min_value=0.0, max_value=1.0, format="%.1f%%"
            ),
        },
    )

st.divider()

# ---- 3. trend ---------------------------------------------------------
st.subheader("Trend")
district = st.selectbox("District", sorted(data["district"].unique()))
window = data[
    (data["district"] == district)
    & (data["date"] >= pd.Timestamp(selected) - pd.Timedelta(days=45))
    & (data["date"] <= pd.Timestamp(selected) + pd.Timedelta(days=15))
].set_index("date")

t1, t2 = st.columns(2)
t1.line_chart(window[["risk"]], height=220)
t2.line_chart(window[["rainfall_mm", "river_level_m"]], height=220)

# ---- 4. explanation ---------------------------------------------------
st.subheader("Why this risk level?")
try:
    from src.models.explain import explain_one

    expl = explain_one(
        district, pd.Timestamp(selected).strftime("%Y-%m-%d"),
        MODEL_NAME, HORIZON, top_n=6,
    )
    expl["effect"] = expl["contribution"].apply(
        lambda v: "↑ increases risk" if v > 0 else "↓ decreases risk"
    )
    st.dataframe(
        expl[["feature", "value", "contribution", "effect"]],
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "SHAP contributions for this district-day. A risk score an officer "
        "cannot interrogate will not be trusted — and should not be."
    )
except Exception as exc:  # noqa: BLE001
    st.info(f"Explanation unavailable: {exc}")

# ---- 5. SOS queue placeholder ----------------------------------------
st.divider()
st.subheader("Citizen SOS queue")
st.info(
    "**Week 5–6 module.** Whisper transcribes voice reports, the NLP layer "
    "classifies incident type and severity, and clustered reports appear "
    "here ranked P1–P3 — and feed back into the fusion layer as "
    "corroborating evidence for the regions above.",
    icon="🎙️",
)
