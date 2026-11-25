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
  persistence     naive baseline
  climatology     day-of-year mean, fitted on train only
  seasonal_naive  y_hat(t+h) = y(t+h-365): last year's value on the same
                  calendar day. On a strongly monsoonal series this is a
                  much harder baseline than climatology, because it keeps
                  the shape of a real year instead of averaging it away.
  ridge_ar        direct multi-horizon ridge regression on the last 14
                  lags plus day-of-year harmonics, one global model per
                  horizon, per-district standardisation fitted on train
                  only. The cheap learned model: if ARIMA or an LSTM
                  cannot beat this, they are not earning their complexity.
  arima           statsmodels SARIMAX, fitted once per district on train
                  then advanced through the test period WITHOUT refitting,
                  so each forecast origin only ever sees past observations
  lstm            PyTorch, multivariate input window -> direct h-step
                  vector output, one global model across districts with
                  per-district standardisation fitted on train only

WHY ARIMA USED TO NEVER FINISH
------------------------------
The first version advanced the state with `append(..., refit=False)` once
per day. `append` re-filters the entire series from the beginning on every
call, so walking 3,287 test days per district costs O(n^2) Kalman passes —
it ran for over an hour on 22 districts without completing. `extend`
continues from the existing filtered state instead, which is what the
walk-forward loop actually needs: the same forecasts, in about a second
per district. Origins are additionally strided (default: weekly) because
daily origins on a smooth series add cost without adding information.

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

def run_arima(panel: pd.DataFrame, target: str, horizons: list[int],
              stride: int | None = None) -> pd.DataFrame:
    """Fit once per district on train, then walk forward without refitting.

    The state is advanced with `extend`, which continues from the existing
    filtered state, rather than `append`, which re-filters the whole series
    on every call (see the module docstring: that is why the first version
    never finished). The forecast made at origin t has only ever seen
    observations up to t either way.

    Refitting the parameters at every origin would be more rigorous and is
    far too slow; fitting once on train and holding the parameters fixed is
    the standard compromise and is stated as such in the report.
    """
    if not HAS_SM:
        print("[forecast] statsmodels missing — skipping ARIMA")
        return pd.DataFrame()

    stride = int(stride or max(horizons))
    train_end, _ = split_dates()
    order = tuple(CFG.forecast.arima_order)
    h_max = max(horizons)
    rows = []

    for district, grp in panel.groupby("district"):
        grp = grp.sort_values("date").reset_index(drop=True)
        series = grp.set_index("date")[target].astype(float)
        series.index = pd.DatetimeIndex(series.index).to_period("D")

        train = series[series.index <= train_end.to_period("D")]
        rest = series[series.index > train_end.to_period("D")]
        if len(train) < 100 or len(rest) < h_max + 1:
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
        n_origins = 0
        for i in range(0, len(rest_values) - h_max, stride):
            fc = np.asarray(res.forecast(steps=h_max))
            origin_date = rest_index[i].to_timestamp()
            for h in horizons:
                rows.append({
                    "district": district,
                    "date": origin_date,
                    "horizon": h,
                    "y_true": float(rest_values[i + h]),
                    "y_pred": float(fc[h - 1]),
                })
            n_origins += 1
            # Advance the state by the next `stride` true observations.
            block = rest.iloc[i : i + stride]
            if block.empty:
                break
            try:
                res = res.extend(block)
            except (AttributeError, NotImplementedError):  # pragma: no cover
                res = res.append(block, refit=False)

        print(f"[forecast]   ARIMA {district}: {n_origins} origins")

    return pd.DataFrame(rows)


# ======================================================================
# Seasonal naive and ridge AR — the cheap models that must be beaten
# ======================================================================

def seasonal_naive(panel: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Last year's value on the same calendar day, 365 days back."""
    rows = []
    for district, grp in panel.groupby("district"):
        grp = grp.sort_values("date").reset_index(drop=True)
        y = grp[target].astype(float).to_numpy()
        dates = grp["date"].to_numpy()
        lag = 365
        for i in range(lag - horizon, len(y) - horizon):
            src = i + horizon - lag
            if src < 0:
                continue
            rows.append({
                "district": district,
                "date": pd.Timestamp(dates[i]),
                "y_true": float(y[i + horizon]),
                "y_pred": float(y[src]),
            })
    return pd.DataFrame(rows)


def run_ridge_ar(panel: pd.DataFrame, target: str, horizons: list[int],
                 n_lags: int = 14) -> pd.DataFrame:
    """Direct multi-horizon ridge regression on lags + seasonality.

    One model per horizon, fitted on the training period only, shared
    across districts after per-district standardisation whose mean and
    standard deviation come from the training period only. Predictions are
    returned on the original scale.
    """
    from sklearn.linear_model import Ridge

    train_end, val_end = split_dates()
    rows = []

    frames = []
    for district, grp in panel.groupby("district"):
        grp = grp.sort_values("date").reset_index(drop=True)
        g = grp[["district", "date", target]].copy()
        tr = g[g["date"] <= train_end][target].astype(float)
        mu = float(tr.mean())
        sd = float(tr.std()) or 1.0
        g["z"] = (g[target].astype(float) - mu) / sd
        g["_mu"], g["_sd"] = mu, sd
        for L in range(n_lags):
            g[f"lag{L}"] = g["z"].shift(L)
        doy = g["date"].dt.dayofyear
        g["sin1"] = np.sin(2 * np.pi * doy / 365.25)
        g["cos1"] = np.cos(2 * np.pi * doy / 365.25)
        g["sin2"] = np.sin(4 * np.pi * doy / 365.25)
        g["cos2"] = np.cos(4 * np.pi * doy / 365.25)
        frames.append(g)

    data = pd.concat(frames, ignore_index=True)
    feats = [f"lag{L}" for L in range(n_lags)] + ["sin1", "cos1", "sin2", "cos2"]

    for h in horizons:
        work = data.copy()
        work["y_future"] = work.groupby("district")["z"].shift(-h)
        work = work.dropna(subset=feats + ["y_future"])
        tr = work[work["date"] <= train_end]
        te = work[work["date"] > val_end]
        if tr.empty or te.empty:
            continue

        model = Ridge(alpha=1.0)
        model.fit(tr[feats], tr["y_future"])
        pred_z = model.predict(te[feats])

        rows.append(pd.DataFrame({
            "district": te["district"].to_numpy(),
            "date": te["date"].to_numpy(),
            "horizon": h,
            "y_true": te["y_future"].to_numpy() * te["_sd"].to_numpy() + te["_mu"].to_numpy(),
            "y_pred": pred_z * te["_sd"].to_numpy() + te["_mu"].to_numpy(),
        }))
        print(f"[forecast]   ridge_ar h={h}: {len(tr):,} train rows, {len(te):,} test rows")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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

def matched_comparison(preds: dict[str, pd.DataFrame], target: str,
                       allow_mape: bool) -> pd.DataFrame:
    """Re-score every model on the rows ALL of them predicted.

    ARIMA is evaluated at strided forecast origins and only for districts
    whose fit converged, so its row count is smaller than the baselines'.
    Comparing MAE across different row sets is not a comparison at all —
    a model evaluated on an easier subset wins for free. This restricts
    every model to the intersection of (district, date, horizon) triples
    and scores again, so the headline skill numbers are like for like.
    """
    keys = None
    for df in preds.values():
        k = set(zip(df["district"], pd.to_datetime(df["date"]), df["horizon"]))
        keys = k if keys is None else (keys & k)
    if not keys:
        return pd.DataFrame()

    rows = []
    for name, df in preds.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        mask = [tuple(t) in keys for t in
                zip(d["district"], d["date"], d["horizon"])]
        d = d[mask]
        for h, part in d.groupby("horizon"):
            rows.append({"model": name, "horizon": int(h),
                         **score(part["y_true"], part["y_pred"], allow_mape)})

    table = pd.DataFrame(rows)
    base = (table[table["model"] == "persistence"]
            .set_index("horizon")["mae"].to_dict())
    table["skill_vs_persistence"] = table.apply(
        lambda r: 1 - r["mae"] / base[r["horizon"]] if base.get(r["horizon"]) else np.nan,
        axis=1,
    )
    return table.sort_values(["horizon", "mae"]).reset_index(drop=True)


def run(target: str | None = None, use_lstm: bool = True,
        arima_stride: int | None = None) -> pd.DataFrame:
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
    for name, fn in (("persistence", persistence), ("climatology", climatology),
                     ("seasonal_naive", seasonal_naive)):
        frames = []
        for h in horizons:
            out = fn(panel, target, h)
            out = out[out["date"] > val_end].assign(horizon=h)
            frames.append(out)
        preds[name] = pd.concat(frames, ignore_index=True)
        print(f"[forecast] {name} done")

    # --- ridge AR (the cheap learned model) ----------------------------
    ridge = run_ridge_ar(panel, target, horizons)
    if len(ridge):
        preds["ridge_ar"] = ridge
        print("[forecast] ridge_ar done")

    # --- ARIMA ---------------------------------------------------------
    arima = run_arima(panel, target, horizons, stride=arima_stride)
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

    matched = matched_comparison(preds, target, allow_mape)

    out_dir = ROOT / CFG.outputs.reports
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"forecast_comparison_{target}.csv", index=False)
    if len(matched):
        matched.to_csv(out_dir / f"forecast_matched_{target}.csv", index=False)
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
    if len(matched):
        print("\nSAME ROWS FOR EVERY MODEL (intersection of forecast origins)")
        print(matched[cols].to_string(index=False,
                                      float_format=lambda x: f"{x:.4f}"))
    print("\nskill > 0 beats persistence · skill < 0 is worse than doing nothing")
    if loaders.is_fixture():
        print("\n*** SYNTHETIC FIXTURE — DO NOT REPORT THESE NUMBERS ***")
    print("=" * 78 + "\n")

    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--arima-stride", type=int, default=None,
                    help="days between forecast origins (default: max horizon)")
    args = ap.parse_args()
    run(target=args.target, use_lstm=not args.no_lstm,
        arima_stride=args.arima_stride)
