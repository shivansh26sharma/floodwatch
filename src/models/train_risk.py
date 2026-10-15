"""
Flood-risk classification: train and compare three models.

  1. Logistic Regression  — the baseline you must beat. If the fancy
                            models don't beat it, say so in the report.
                            That is a finding, not a failure.
  2. Random Forest
  3. XGBoost

All three are trained on the SAME chronological split and the SAME
features, so the comparison is fair. Probabilities are calibrated on
the validation split (see calibrate.py) because an uncalibrated
probability cannot be acted on by a human.

Usage
-----
    python -m src.models.train_risk                # horizon 1
    python -m src.models.train_risk --horizon 3
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CFG, ROOT
from src.data import loaders
from src.features.build_features import TARGET, chronological_split, feature_columns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def build_models() -> dict:
    m = CFG.models
    models = {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=m.logistic.max_iter,
                class_weight=m.logistic.class_weight,
                random_state=m.random_state,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=m.random_forest.n_estimators,
            max_depth=m.random_forest.max_depth,
            min_samples_leaf=m.random_forest.min_samples_leaf,
            class_weight=m.random_forest.class_weight,
            random_state=m.random_state,
            n_jobs=-1,
        ),
    }

    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=m.xgboost.n_estimators,
            max_depth=m.xgboost.max_depth,
            learning_rate=m.xgboost.learning_rate,
            subsample=m.xgboost.subsample,
            colsample_bytree=m.xgboost.colsample_bytree,
            random_state=m.random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        print("[train] xgboost not installed — skipping. pip install xgboost")

    return models


def load_features(horizon: int) -> tuple[pd.DataFrame, list[str]]:
    path = ROOT / "data/processed" / f"features_h{horizon}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Build it first:\n"
            f"    python -m src.features.build_features --horizon {horizon}"
        )
    df = pd.read_parquet(path)
    return df, feature_columns(df)


def train(horizon: int = 1, save: bool = True) -> dict:
    df, feats = load_features(horizon)
    splits = chronological_split(df)

    # Class imbalance handling for XGBoost is via scale_pos_weight,
    # computed from the training split only.
    y_train = splits["train"][TARGET]
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    print(f"[train] features={len(feats)}  horizon={horizon}d")
    for name, part in splits.items():
        print(f"[train]   {name:6s} {len(part):>7,} rows  "
              f"{part[TARGET].mean():>6.2%} positive  "
              f"{part['date'].min().date()} .. {part['date'].max().date()}")

    # ---- the sparse-label guard ---------------------------------------
    # Real flood archives record only LARGE events, so a split can easily
    # contain zero positives — at which point sklearn raises a cryptic
    # "needs samples of at least 2 classes". Fail here instead, with the
    # actual numbers and the actual fix.
    empty = {
        name: int(part[TARGET].sum())
        for name, part in splits.items()
        if part[TARGET].sum() == 0
    }
    if empty:
        counts = "\n".join(
            f"    {n:6s} {int(p[TARGET].sum()):>4} positives in {len(p):,} rows "
            f"({p['date'].min().date()} .. {p['date'].max().date()})"
            for n, p in splits.items()
        )
        raise ValueError(
            f"No flood days in split(s): {sorted(empty)}. Nothing to learn from.\n"
            f"{counts}\n\n"
            f"This is the normal failure mode with real DFO labels — large\n"
            f"events are sparse and can miss a window entirely. Fixes, best\n"
            f"first:\n"
            f"  1. Widen project.start_date in config.yaml (DFO covers 1985+,\n"
            f"     and ERA5/GloFAS go back just as far) and re-download.\n"
            f"  2. Move split.train_end / split.val_end so each window spans\n"
            f"     some recorded events — keep the anchor event in test.\n"
            f"  3. Add districts in project.districts to catch more events.\n"
            f"Do NOT shuffle the split to fix this: that leaks the future."
        )

    trained, predictions = {}, {}
    for name, model in build_models().items():
        if name == "xgboost":
            model.set_params(scale_pos_weight=pos_weight)

        model.fit(splits["train"][feats], y_train)
        trained[name] = model

        predictions[name] = {
            part: model.predict_proba(splits[part][feats])[:, 1]
            for part in splits
        }
        print(f"[train] fitted {name}")

    if save:
        out_dir = ROOT / CFG.outputs.models
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, model in trained.items():
            with open(out_dir / f"risk_{name}_h{horizon}.pkl", "wb") as fh:
                pickle.dump({"model": model, "features": feats, "horizon": horizon}, fh)

        meta = {
            "horizon": horizon,
            "n_features": len(feats),
            "features": feats,
            "data_source": CFG.data.source,
            "synthetic_fixture": loaders.is_fixture(),
            "split": {k: [str(v["date"].min().date()), str(v["date"].max().date())]
                      for k, v in splits.items()},
            "positive_rate": {k: float(v[TARGET].mean()) for k, v in splits.items()},
            "scale_pos_weight": pos_weight,
        }
        with open(out_dir / f"risk_meta_h{horizon}.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[train] saved {len(trained)} models to {out_dir}")

    if loaders.is_fixture():
        print("[train] *** SYNTHETIC FIXTURE — these numbers mean nothing ***")

    return {"models": trained, "predictions": predictions,
            "splits": splits, "features": feats, "horizon": horizon}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()
    train(horizon=args.horizon)
