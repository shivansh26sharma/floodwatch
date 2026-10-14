"""
Time-series forecasting: river level / rainfall, 1-7 days ahead.

THE POINT OF THIS MODULE IS NOT "WE USED AN LSTM".
---------------------------------------------------
It is to answer one question honestly: **can anything beat persistence?**

River levels are highly autocorrelated. "Tomorrow's level equals today's
level" is a brutally strong baseline, and a large share of published
hydrological deep-learning results quietly fail to beat it. So every
model here is scored against two naive baselines:

    PERSISTENCE   y_hat(t+h) = y(t)              the one that matters
    CLIMATOLOGY   y_hat(t+h) = mean for that day-of-year (train only)

and the headline number is the **skill score**:

    skill = 1 - MAE_model / MAE_persistence

    > 0   better than doing nothing
    = 0   exactly as good as doing nothing
    < 0   worse than doing nothing   <- report this if it happens

If your LSTM gets skill < 0 at h=1 and > 0 at h=7, that is a real and
interesting finding: persistence wins short-horizon, the model earns its
keep further out. Say so. It is a better report than a hidden failure.

MODELS
------
  persistence   naive baseline
  climatology   day-of-year mean, fitted on train only
  arima         statsmodels SARIMAX, fitted once on train then advanced
                through the test period with append(refit=False) so each
                forecast origin only ever sees past observations
  lstm          PyTorch, multivariate input window -> direct h-step vector
                output, one global model across districts with per-district
                standardisation fitted on train only

A NOTE ON MAPE
--------------
MAPE is in the proposal and is reported for river level, but it is
undefined wherever the true value is zero — and rainfall is zero on most
days. So for rainfall we report MAE/RMSE and sMAPE instead, and say why.
Blindly reporting MAPE on zero-inflated rainfall produces infinities or
absurd percentages, which is a classic way to lose marks.

Usage
-----
    python -m src.models.forecast                      # river level
    python -m src.models.forecast --target rainfall_mm
    python -m src.models.forecast --no-lstm            # skip the NN
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from src.config import CFG, ROOT
from src.data import loaders

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SM = True
except ImportError:  # pragma: no cover
    HAS_SM = False


# ======================================================================
# Data
# ======================================================================

def load_panel(target: str) -> pd.DataFrame:
    path = ROOT / CFG.data.processed
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run:\n    python -m src.data.build_dataset"
        )
    cols = ["district", "date", target] + [
        c for c in CFG.forecast.exog if c != target
    ]
    df = pd.read_parquet(path)[list(dict.fromkeys(cols))]
    return df.sort_values(["district", "date"]).reset_index(drop=True)


def split_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(CFG.split.train_end), pd.Timestamp(CFG.split.val_end)


# ======================================================================
# Metrics
# ======================================================================

def _smape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = (np.abs(y) + np.abs(yhat)) / 2
    mask = denom > 1e-9
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100)


def score(y: np.ndarray, yhat: np.ndarray, allow_mape: bool) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[ok], yhat[ok]
    if len(y) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "smape": np.nan, "n": 0}

    err = y - yhat
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "smape": _smape(y, yhat),
        "n": int(len(y)),
    }
    if allow_mape:
        nz = np.abs(y) > 1e-6
        out["mape"] = (
            float(np.mean(np.abs(err[nz] / y[nz])) * 100) if nz.sum() else np.nan
        )
    else:
        # Undefined on zero-inflated series; sMAPE is reported instead.
        out["mape"] = np.nan
    return out


# ======================================================================
# Baselines
# ======================================================================

def persistence(panel: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """y_hat(t+h) = y(t). The baseline every model must beat."""
    df = panel.copy()
    g = df.groupby("district", group_keys=False)
    df["y_true"] = g[target].shift(-horizon)
    df["y_pred"] = df[target]
    return df.dropna(subset=["y_true"])[["district", "date", "y_true", "y_pred"]]


def climatology(panel: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Day-of-year mean per district, fitted on the training period only."""
    train_end, _ = split_dates()
    train = panel[panel["date"] <= train_end].copy()
    train["doy"] = train["date"].dt.dayofyear
    clim = (
        train.groupby(["district", "doy"], as_index=False)[target]
        .mean().rename(columns={target: "clim"})
    )
    clim["clim"] = (
        clim.sort_values(["district", "doy"])
        .groupby("district", group_keys=False)["clim"]
        .apply(lambda s: s.rolling(15, center=True, min_periods=1).mean())
    )

    df = panel.copy()
    df["y_true"] = df.groupby("district", group_keys=False)[target].shift(-horizon)
    df["doy"] = (df["date"] + pd.Timedelta(days=horizon)).dt.dayofyear
    df = df.merge(clim, on=["district", "doy"], how="left")
    df["y_pred"] = df["clim"]
    return df.dropna(subset=["y_true", "y_pred"])[
        ["district", "date", "y_true", "y_pred"]
    ]


# ======================================================================
# ARIMA
# ======================================================================

def run_arima(panel: pd.DataFrame, target: str, horizons: list[int]) -> pd.DataFrame:
    """Fit once per district on train, then walk forward without refitting.

    `append(..., refit=False)` extends the filtered state with each new
    observation, so the forecast made at origin t has only ever seen data
    up to t. Refitting at every origin would be more rigorous and is far
    too slow for 8 districts x 2 years; this is the standard compromise
    and you should say so in the report.
    """
    if not HAS_SM:
        print("[forecast] statsmodels missing — skipping ARIMA")
        return pd.DataFrame()

    train_end, _ = split_dates()
    order = tuple(CFG.forecast.arima_order)
    rows = []

    for district, grp in panel.groupby("district"):
        grp = grp.sort_values("date").reset_index(drop=True)
        series = grp.set_index("date")[target].astype(float)
        series.index = pd.DatetimeIndex(series.index).to_period("D")

        train = series[series.index <= train_end.to_period("D")]
        rest = series[series.index > train_end.to_period("D")]
        if len(train) < 100 or len(rest) < max(horizons) + 1:
            continue

        try:
            res = SARIMAX(
                train, order=order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[forecast]   ARIMA failed for {district}: {exc}")
            continue

        rest_values = rest.to_numpy()
        rest_index = rest.index
        for i in range(len(rest_values) - max(horizons)):
            fc = res.forecast(steps=max(horizons))
            origin_date = rest_index[i].to_timestamp()
            for h in horizons:
                rows.append({
                    "district": district,
                    "date": origin_date,
                    "horizon": h,
                    "y_true": float(rest_values[i + h]),
                    "y_pred": float(np.asarray(fc)[h - 1]),
                })
            # advance state by one true observation, no refit
            res = res.append(rest.iloc[i : i + 1], refit=False)

        print(f"[forecast]   ARIMA {district} done")

    return pd.DataFrame(rows)


# ======================================================================
# LSTM
# ======================================================================

class LSTMForecaster(nn.Module if HAS_TORCH else object):
    """Multivariate window -> direct multi-horizon vector output.

    Direct (not recursive) multi-step: the network outputs all horizons at
    once, so errors do not compound across steps and each horizon is
    trained against its own true value.
    """

    def __init__(self, n_features: int, n_out: int):
        super().__init__()
        c = CFG.forecast.lstm
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=c.hidden, num_layers=c.layers,
            batch_first=True, dropout=c.dropout if c.layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(c.hidden, c.hidden // 2), nn.ReLU(), nn.Linear(c.hidden // 2, n_out)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _windows(values: np.ndarray, target_idx: int, window: int, horizons: list[int]):
    """Build (X, y) where y holds the true value at each horizon."""
    max_h = max(horizons)
    X, y, ends = [], [], []
    for i in range(window, len(values) - max_h):
        X.append(values[i - window : i])
        y.append([values[i + h - 1, target_idx] for h in horizons])
        ends.append(i)
    if not X:
        return np.empty((0, window, values.shape[1])), np.empty((0, len(horizons))), []
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), ends


def run_lstm(panel: pd.DataFrame, target: str, horizons: list[int]) -> pd.DataFrame:
    if not HAS_TORCH:
        print("[forecast] torch missing — skipping LSTM")
        return pd.DataFrame()

    c = CFG.forecast.lstm
    train_end, val_end = split_dates()
    feat_cols = [target] + [x for x in CFG.forecast.exog if x != target]
    torch.manual_seed(CFG.models.random_state)

    Xtr, ytr, Xva, yva, Xte, yte, meta_te = [], [], [], [], [], [], []

    for district, grp in panel.groupby("district"):
        grp = grp.sort_values("date").reset_index(drop=True)
        vals = grp[feat_cols].to_numpy(dtype=float)
        dates = grp["date"].to_numpy()

        # Standardise per district using TRAIN STATISTICS ONLY.
        tr_mask = grp["date"] <= train_end
        mu = vals[tr_mask.to_numpy()].mean(axis=0)
        sd = vals[tr_mask.to_numpy()].std(axis=0) + 1e-8
        z = (vals - mu) / sd

        X, y, ends = _windows(z, 0, c.window, horizons)
        if len(X) == 0:
            continue
        end_dates = pd.to_datetime(dates[ends])

        tr = end_dates <= train_end
        va = (end_dates > train_end) & (end_dates <= val_end)
        te = end_dates > val_end

        Xtr.append(X[tr]); ytr.append(y[tr])
        Xva.append(X[va]); yva.append(y[va])
        Xte.append(X[te]); yte.append(y[te])
        meta_te.append(pd.DataFrame({
            "district": district,
            "date": end_dates[te],
            "mu": mu[0], "sd": sd[0],
        }))

    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    Xva, yva = np.concatenate(Xva), np.concatenate(yva)
    Xte, yte = np.concatenate(Xte), np.concatenate(yte)
    meta_te = pd.concat(meta_te, ignore_index=True)

    print(f"[forecast]   LSTM windows: train {len(Xtr):,} / val {len(Xva):,} "
          f"/ test {len(Xte):,}")

    model = LSTMForecaster(Xtr.shape[2], len(horizons))
    opt = torch.optim.Adam(model.parameters(), lr=c.lr)
    lossf = nn.L1Loss()                      # MAE: robust to flood spikes

    tXtr, tytr = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    tXva, tyva = torch.from_numpy(Xva), torch.from_numpy(yva)

    best, best_state, bad = float("inf"), None, 0
    n = len(tXtr)
    for epoch in range(c.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, c.batch_size):
            idx = perm[i : i + c.batch_size]
            opt.zero_grad()
            loss = lossf(model(tXtr[idx]), tytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_mae = lossf(model(tXva), tyva).item()

        if val_mae < best - 1e-5:
            best, bad = val_mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c.patience:
                print(f"[forecast]   early stop at epoch {epoch + 1} "
                      f"(best val MAE {best:.4f} in z-units)")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_z = model(torch.from_numpy(Xte)).numpy()

    # Back to real units using each district's own train statistics.
    rows = []
    mu_te = meta_te["mu"].to_numpy()[:, None]
    sd_te = meta_te["sd"].to_numpy()[:, None]
    pred = pred_z * sd_te + mu_te
    true = yte * sd_te + mu_te

    for j, h in enumerate(horizons):
        rows.append(pd.DataFrame({
            "district": meta_te["district"],
            "date": meta_te["date"],
            "horizon": h,
            "y_true": true[:, j],
            "y_pred": pred[:, j],
        }))
    return pd.concat(rows, ignore_index=True)


# ======================================================================
# Orchestration
# ======================================================================

def run(target: str | None = None, use_lstm: bool = True) -> pd.DataFrame:
    target = target or CFG.forecast.target
    horizons = list(CFG.forecast.horizons)
    allow_mape = target != "rainfall_mm"          # see module docstring

    panel = load_panel(target)
    _, val_end = split_dates()

    print(f"[forecast] target={target}  horizons={horizons}")
    if not allow_mape:
        print("[forecast] target is zero-inflated — reporting sMAPE, not MAPE")

    preds: dict[str, pd.DataFrame] = {}

    # --- baselines (evaluated on the test period only) -----------------
    for name, fn in (("persistence", persistence), ("climatology", climatology)):
        frames = []
        for h in horizons:
            out = fn(panel, target, h)
            out = out[out["date"] > val_end].assign(horizon=h)
            frames.append(out)
        preds[name] = pd.concat(frames, ignore_index=True)
        print(f"[forecast] {name} done")

    # --- ARIMA ---------------------------------------------------------
    arima = run_arima(panel, target, horizons)
    if len(arima):
        preds["arima"] = arima[arima["date"] > val_end]

    # --- LSTM ----------------------------------------------------------
    if use_lstm:
        lstm = run_lstm(panel, target, horizons)
        if len(lstm):
            preds["lstm"] = lstm

    # --- scoring --------------------------------------------------------
    rows = []
    for name, df in preds.items():
        for h in horizons:
            part = df[df["horizon"] == h]
            if part.empty:
                continue
            rows.append({
                "model": name, "horizon": h,
                **score(part["y_true"], part["y_pred"], allow_mape),
            })
    table = pd.DataFrame(rows)

    # Skill score vs persistence — the headline.
    base = (
        table[table["model"] == "persistence"]
        .set_index("horizon")["mae"].to_dict()
    )
    table["skill_vs_persistence"] = table.apply(
        lambda r: 1 - r["mae"] / base[r["horizon"]] if base.get(r["horizon"]) else np.nan,
        axis=1,
    )
    table = table.sort_values(["horizon", "mae"]).reset_index(drop=True)

    out_dir = ROOT / CFG.outputs.reports
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"forecast_comparison_{target}.csv", index=False)
    for name, df in preds.items():
        df.to_csv(out_dir / f"forecast_preds_{name}_{target}.csv", index=False)

    with open(out_dir / f"forecast_summary_{target}.json", "w") as fh:
        json.dump({
            "target": target,
            "horizons": horizons,
            "synthetic_fixture": loaders.is_fixture(),
            "mape_reported": allow_mape,
            "results": table.to_dict(orient="records"),
        }, fh, indent=2, default=float)

    print("\n" + "=" * 78)
    print(f"FORECAST COMPARISON — {target}")
    print("=" * 78)
    cols = ["model", "horizon", "mae", "rmse",
            ("mape" if allow_mape else "smape"), "skill_vs_persistence"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nskill > 0 beats persistence · skill < 0 is worse than doing nothing")
    if loaders.is_fixture():
        print("\n*** SYNTHETIC FIXTURE — DO NOT REPORT THESE NUMBERS ***")
    print("=" * 78 + "\n")

    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--no-lstm", action="store_true")
    args = ap.parse_args()
    run(target=args.target, use_lstm=not args.no_lstm)
