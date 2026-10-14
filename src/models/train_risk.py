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
    XGB_ERROR = ""
except Exception as _exc:  # noqa: BLE001  pragma: no cover
    # Deliberately broad. xgboost can be pip-installed yet fail to load its
    # native libxgboost.dylib (macOS without libomp), which raises
    # XGBoostError, not ImportError. Catching only ImportError here turned a
    # graceful skip into a crash of the whole pipeline.
    HAS_XGB = False
    XGB_ERROR = str(_exc).strip().splitlines()[0] if str(_exc).strip() else type(_exc).__name__


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
        print(f"[train] xgboost unavailable — skipping it. Reason: {XGB_ERROR}")
        if "libomp" in XGB_ERROR or "libxgboost" in XGB_ERROR:
            print("[train]   macOS fix:  brew install libomp")
        else:
            print("[train]   fix:  pip install xgboost")

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
