"""
One-call pipeline bootstrap.

On a laptop you run the steps yourself via scripts/run_pipeline.sh. On a
fresh cloud deploy there is no such chance: Streamlit Community Cloud
clones the repo and immediately starts the app, and the processed parquet
and trained models are gitignored (correctly — they are build artifacts,
not source).

So the app calls `ensure_artifacts()` on startup. If the artifacts are
missing it builds them from the committed fixture, which takes ~20s once
and is then cached. If they are present it returns instantly.

This is also why the fixture is committed while data/processed is not:
the fixture is the *seed* that makes a fresh clone self-healing.
"""

from pathlib import Path

from src.config import CFG, ROOT


def artifact_status(horizon: int = 1) -> dict[str, bool]:
    return {
        "fixture": (ROOT / CFG.data.fixture.path / "rainfall.csv").exists(),
        "dataset": (ROOT / CFG.data.processed).exists(),
        "features": (ROOT / "data/processed" / f"features_h{horizon}.parquet").exists(),
        "model": (ROOT / CFG.outputs.models / f"risk_random_forest_h{horizon}.pkl").exists(),
        "summary": (ROOT / CFG.outputs.reports / f"summary_h{horizon}.json").exists(),
    }


def ensure_artifacts(horizon: int = 1, verbose: bool = True) -> dict[str, bool]:
    """Build anything missing. Safe to call repeatedly."""
    status = artifact_status(horizon)

    def log(msg: str) -> None:
        if verbose:
            print(f"[bootstrap] {msg}")

    if not status["fixture"]:
        if CFG.data.source == "real":
            raise RuntimeError(
                "config says data.source=real but no real data is present. "
                "Place the downloads in data/raw/ and implement the loaders."
            )
        log("generating fixture")
        from src.data.make_fixture import write
        write()

    if not status["dataset"]:
        log("building dataset")
        from src.data.build_dataset import build
        build()

    if not status["features"]:
        log(f"building features (horizon {horizon})")
        from src.features.build_features import build as build_features
        build_features(horizon=horizon)

    if not (status["model"] and status["summary"]):
        # evaluate() trains and then writes the threshold summary the
        # dashboard needs, so one call covers both.
        log("training models and evaluating (first run only, ~20s)")
        from src.models.evaluate import run as evaluate_run
        evaluate_run(horizon=horizon)

    final = artifact_status(horizon)
    log("ready" if all(final.values()) else f"incomplete: {final}")
    return final


if __name__ == "__main__":
    ensure_artifacts()
