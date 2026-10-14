"""
Explainability with SHAP.

Two outputs, for two different audiences:

  GLOBAL  — which features drive the model overall. This goes in your
            report and answers "what did the model learn?"

  LOCAL   — for one district on one date, which features pushed the risk
            up or down. This goes in the dashboard, so an officer sees
            *why* their district was flagged and can sanity-check it.
            A risk score nobody can interrogate won't be trusted, and
            shouldn't be.

Only tree models are explained with TreeExplainer (fast, exact).
The logistic baseline is explained by its coefficients, which is both
cheaper and more honest for a linear model.

Usage
-----
    python -m src.models.explain --horizon 1 --model random_forest
"""

import argparse
import pickle

import numpy as np
import pandas as pd

from src.config import CFG, ROOT
from src.features.build_features import TARGET, chronological_split

try:
    import shap
    HAS_SHAP = True
except Exception:  # noqa: BLE001  pragma: no cover
    # Broad on purpose: see the note in train_risk.py. A dependency that
    # imports but cannot load its native library must degrade, not crash.
    HAS_SHAP = False


def load_model(name: str, horizon: int):
    path = ROOT / CFG.outputs.models / f"risk_{name}_h{horizon}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Train first:\n"
            f"    python -m src.models.train_risk --horizon {horizon}"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """SHAP on 30k rows is slow and unnecessary; a sample is plenty."""
    if len(df) <= n:
        return df
    return df.sample(n, random_state=seed)


def global_importance(
    name: str = "random_forest", horizon: int = 1, n_sample: int = 2000
) -> pd.DataFrame:
    bundle = load_model(name, horizon)
    model, feats = bundle["model"], bundle["features"]

    df = pd.read_parquet(ROOT / "data/processed" / f"features_h{horizon}.parquet")
    test = chronological_split(df)["test"]
    X = _sample(test[feats], n_sample)

    if name == "logistic":
        # Linear model: standardised coefficients are the honest explanation.
        clf = model.named_steps["clf"]
        imp = pd.DataFrame({
            "feature": feats,
            "importance": np.abs(clf.coef_[0]),
            "direction": np.sign(clf.coef_[0]),
        })
    else:
        if not HAS_SHAP:
            raise ImportError("pip install shap")
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        # Binary classifiers may return a list (one array per class).
        if isinstance(values, list):
            values = values[1]
        # Newer SHAP returns (n, features, classes) for some models.
        if values.ndim == 3:
            values = values[:, :, 1]
        imp = pd.DataFrame({
            "feature": feats,
            "importance": np.abs(values).mean(axis=0),
            "direction": np.sign((values * (X.values - X.values.mean(0))).mean(axis=0)),
        })

    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

    out = ROOT / CFG.outputs.reports / f"shap_global_{name}_h{horizon}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out, index=False)

    print(f"\nTop 15 features — {name} (horizon {horizon}d)")
    print(imp.head(15).to_string(index=False))
    print(f"\nwrote {out}")
    return imp


def explain_one(
    district: str, date: str, name: str = "random_forest",
    horizon: int = 1, top_n: int = 6,
) -> pd.DataFrame:
    """Local explanation for a single district-day — used by the dashboard."""
    bundle = load_model(name, horizon)
    model, feats = bundle["model"], bundle["features"]

    df = pd.read_parquet(ROOT / "data/processed" / f"features_h{horizon}.parquet")
    row = df[(df["district"] == district) & (df["date"] == pd.Timestamp(date))]
    if row.empty:
        raise ValueError(f"No row for {district} on {date}")

    X = row[feats]
    proba = float(model.predict_proba(X)[:, 1][0])

    if name == "logistic":
        clf = model.named_steps["clf"]
        scaler = model.named_steps["scale"]
        contrib = clf.coef_[0] * scaler.transform(X)[0]
    else:
        if not HAS_SHAP:
            raise ImportError("pip install shap")
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[1]
        if np.ndim(values) == 3:
            values = values[:, :, 1]
        contrib = np.asarray(values)[0]

    expl = (
        pd.DataFrame({
            "feature": feats,
            "value": X.iloc[0].to_numpy(),
            "contribution": contrib,
        })
        .assign(abs_contribution=lambda d: d["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print(f"\n{district} on {date} — predicted risk {proba:.1%} ({name})")
    print("Top contributing factors:")
    for _, r in expl.iterrows():
        arrow = "increases" if r["contribution"] > 0 else "decreases"
        print(f"  {r['feature']:32s} = {r['value']:>10.2f}  {arrow} risk "
              f"({r['contribution']:+.3f})")
    return expl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--model", default="random_forest")
    ap.add_argument("--district", default=None)
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    if args.district and args.date:
        explain_one(args.district, args.date, args.model, args.horizon)
    else:
        global_importance(args.model, args.horizon)
