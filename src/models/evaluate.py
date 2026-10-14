"""
Evaluation — deliberately NOT just accuracy.

Four things are measured here, in rough order of how much they matter
for a warning system:

  1. LEAD TIME. For each real flood event, how long before onset did the
     model first raise the alarm? A model with 95% accuracy and 40
     minutes of warning is worthless; one with 80% accuracy and 18 hours
     saves a district. This is the headline metric.

  2. COST-SENSITIVE THRESHOLD. Missing a flood is far worse than a false
     alarm, so the operating threshold is chosen to minimise expected
     cost at the configured cost ratio — not to maximise F1, and
     certainly not left at 0.5.

  3. CALIBRATION. If the model says 70%, it should flood ~70% of the
     time. An uncalibrated score cannot be acted on by a human.
     Reported as Brier score plus a reliability curve.

  4. DISCRIMINATION. ROC-AUC, PR-AUC, and the confusion matrix at the
     chosen threshold. Recall is reported first because that is the one
     that corresponds to people not being warned.

Usage
-----
    python -m src.models.evaluate --horizon 1
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

from src.config import CFG, ROOT
from src.data import loaders
from src.features.build_features import TARGET
from src.models.train_risk import train


# ----------------------------------------------------------------------
# 1. Lead time
# ----------------------------------------------------------------------

def find_events(df: pd.DataFrame, label_col: str = "flood_occurred") -> pd.DataFrame:
    """Collapse consecutive flood days into events with an onset date."""
    events = []
    for district, grp in df.sort_values("date").groupby("district"):
        flooded = grp[label_col].to_numpy()
        dates = grp["date"].to_numpy()
        # Onset = a 1 whose previous day was 0 (or the series start).
        onsets = np.flatnonzero((flooded == 1) & (np.r_[0, flooded[:-1]] == 0))
        for i in onsets:
            end = i
            while end + 1 < len(flooded) and flooded[end + 1] == 1:
                end += 1
            events.append({
                "district": district,
                "onset": pd.Timestamp(dates[i]),
                "end": pd.Timestamp(dates[end]),
                "duration_days": end - i + 1,
            })
    return pd.DataFrame(events)


def lead_times(
    df: pd.DataFrame, proba: np.ndarray, threshold: float, max_lookback_days: int = 14
) -> pd.DataFrame:
    """For each event, hours between first threshold crossing and onset.

    A crossing only counts if the alarm stays conceptually 'on' — we take
    the FIRST crossing in an unbroken run of above-threshold days leading
    into the onset, which is what an operator would actually experience.
    Events never predicted get lead_time_hours = NaN (a miss).
    """
    work = df[["district", "date", "flood_occurred"]].copy()
    work["proba"] = proba
    work["alarm"] = work["proba"] >= threshold

    events = find_events(work)
    rows = []
    for _, ev in events.iterrows():
        window = work[
            (work["district"] == ev["district"])
            & (work["date"] < ev["onset"])
            & (work["date"] >= ev["onset"] - pd.Timedelta(days=max_lookback_days))
        ].sort_values("date")

        if window.empty or not window["alarm"].iloc[-1]:
            # No alarm on the day before onset — count as a miss for lead-time
            # purposes even if an earlier isolated spike occurred.
            rows.append({**ev, "lead_time_hours": np.nan, "warned": False})
            continue

        # Walk back through the unbroken run of alarm days.
        run_start = window["date"].iloc[-1]
        for date, alarm in zip(window["date"].iloc[::-1], window["alarm"].iloc[::-1]):
            if not alarm:
                break
            run_start = date

        hours = (ev["onset"] - run_start) / pd.Timedelta(hours=1)
        rows.append({**ev, "lead_time_hours": hours, "warned": True})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. Cost-sensitive threshold
# ----------------------------------------------------------------------

def cost_curve(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Expected cost across thresholds, at the configured cost ratio."""
    c_fn = CFG.evaluation.cost_false_negative
    c_fp = CFG.evaluation.cost_false_positive

    rows = []
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": round(float(t), 3),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "expected_cost": c_fn * fn + c_fp * fp,
        })
    return pd.DataFrame(rows)


def pick_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, pd.DataFrame]:
    curve = cost_curve(y_true, proba)
    best = curve.loc[curve["expected_cost"].idxmin()]
    return float(best["threshold"]), curve


# ----------------------------------------------------------------------
# 3. Calibration
# ----------------------------------------------------------------------

def reliability(y_true: np.ndarray, proba: np.ndarray, bins: int | None = None) -> pd.DataFrame:
    bins = bins or CFG.evaluation.calibration_bins
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(proba, edges) - 1, 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_mid": float((edges[b] + edges[b + 1]) / 2),
            "n": int(mask.sum()),
            "mean_predicted": float(proba[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Full report
# ----------------------------------------------------------------------

def evaluate_model(
    name: str, df: pd.DataFrame, proba_val: np.ndarray,
    df_val: pd.DataFrame, proba_test: np.ndarray,
) -> dict:
    y_val = df_val[TARGET].to_numpy()
    y_test = df[TARGET].to_numpy()

    # Threshold chosen on VALIDATION, then applied unchanged to test.
    # Choosing it on test would be cheating.
    threshold, curve = pick_threshold(y_val, proba_val)

    pred = (proba_test >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )

    lt = lead_times(df, proba_test, threshold)
    warned = lt["warned"].sum() if len(lt) else 0

    # ---- the lead-time sanity check ----------------------------------
    # Lead time alone is gameable: a model that keeps the alarm on all
    # monsoon shows enormous "lead time" and is operationally useless.
    # So we always report it alongside the alarm duty cycle — the share of
    # district-days the alarm is raised — and the ratio between them.
    # A good system has long lead time AND a low duty cycle.
    alarm_rate = float((proba_test >= threshold).mean())

    return {
        "model": name,
        "threshold": threshold,
        "alarm_duty_cycle": alarm_rate,
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "pr_auc": float(average_precision_score(y_test, proba_test)),
        "brier": float(brier_score_loss(y_test, proba_test)),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "expected_cost": float(
            CFG.evaluation.cost_false_negative * fn
            + CFG.evaluation.cost_false_positive * fp
        ),
        "events_total": int(len(lt)),
        "events_warned": int(warned),
        "event_detection_rate": float(warned / len(lt)) if len(lt) else None,
        "median_lead_time_hours": (
            float(lt["lead_time_hours"].median()) if warned else None
        ),
        "mean_lead_time_hours": (
            float(lt["lead_time_hours"].mean()) if warned else None
        ),
        "_cost_curve": curve,
        "_reliability": reliability(y_test, proba_test),
        "_lead_times": lt,
    }


def run(horizon: int = 1) -> pd.DataFrame:
    result = train(horizon=horizon, save=True)
    splits, preds = result["splits"], result["predictions"]

    reports = []
    for name in preds:
        rep = evaluate_model(
            name,
            df=splits["test"], proba_test=preds[name]["test"],
            df_val=splits["val"], proba_val=preds[name]["val"],
        )
        reports.append(rep)

    table = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")} for r in reports
    ])

    out_dir = ROOT / CFG.outputs.reports
    out_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(out_dir / f"model_comparison_h{horizon}.csv", index=False)
    for r in reports:
        stem = f"{r['model']}_h{horizon}"
        r["_cost_curve"].to_csv(out_dir / f"cost_curve_{stem}.csv", index=False)
        r["_reliability"].to_csv(out_dir / f"reliability_{stem}.csv", index=False)
        r["_lead_times"].to_csv(out_dir / f"lead_times_{stem}.csv", index=False)

    with open(out_dir / f"summary_h{horizon}.json", "w") as fh:
        json.dump({
            "synthetic_fixture": loaders.is_fixture(),
            "horizon_days": horizon,
            "cost_ratio_fn_to_fp": (
                CFG.evaluation.cost_false_negative / CFG.evaluation.cost_false_positive
            ),
            "models": [{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in reports],
        }, fh, indent=2)

    # ---- console report ------------------------------------------------
    print("\n" + "=" * 78)
    print(f"MODEL COMPARISON  (horizon = {horizon} day(s) ahead)")
    print("=" * 78)
    cols = ["model", "threshold", "recall", "precision", "roc_auc", "pr_auc",
            "brier", "event_detection_rate", "median_lead_time_hours",
            "alarm_duty_cycle"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nConfusion matrices at the cost-optimal threshold:")
    for r in reports:
        c = r["confusion"]
        print(f"  {r['model']:14s} tp={c['tp']:>5} fp={c['fp']:>5} "
              f"fn={c['fn']:>5} tn={c['tn']:>6}   cost={r['expected_cost']:,.0f}")
    print(f"\nReports written to {out_dir}")
    if loaders.is_fixture():
        print("\n*** SYNTHETIC FIXTURE — DO NOT PUT THESE NUMBERS IN YOUR REPORT ***")
    print("=" * 78 + "\n")

    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()
    run(horizon=args.horizon)
