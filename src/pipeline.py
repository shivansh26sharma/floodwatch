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

import hashlib
import json
from pathlib import Path

from src.config import CFG, ROOT


def _effective_source_for_fingerprint() -> str:
    try:
        from src.data.loaders import effective_source
        return effective_source()
    except Exception:  # noqa: BLE001
        return str(CFG.data.source)


def build_fingerprint() -> str:
    """A hash of every config choice that changes the SHAPE of the artifacts.

    Existence is not freshness. A deploy that already has
    features_h1.parquet from a previous version will happily reuse it even
    though a column was renamed underneath it — which is exactly how the
    hosted dashboard ended up raising KeyError on river_discharge_m3s while
    holding a parquet that still said river_level_m.

    So we stamp the build with a fingerprint and rebuild when it moves.
    Anything that alters column names, the district set, the date range or
    the target definition belongs in here.
    """
    parts = {
        "districts": sorted(CFG.project.districts) if isinstance(
            CFG.project.districts, list) else str(CFG.project.districts),
        "start": str(CFG.project.start_date),
        "end": str(CFG.project.end_date),
        # The EFFECTIVE source, not the configured one — a deploy that
        # falls back to the fixture must not reuse artifacts built from
        # real data, or vice versa.
        "source": _effective_source_for_fingerprint(),
        "target_mode": str(getattr(CFG.features, "target_mode", "onset_window")),
        "target_window": int(getattr(CFG.features, "target_window", 7)),
        "train_end": str(CFG.split.train_end),
        "val_end": str(CFG.split.val_end),
        # Schema version — bump by hand after any column rename.
        "schema": 2,
    }
    blob = json.dumps(parts, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _manifest_path() -> Path:
    return ROOT / "data/processed/build_manifest.json"


def _stored_fingerprint() -> str | None:
    path = _manifest_path()
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh).get("fingerprint")
    except Exception:  # noqa: BLE001
        return None


def _write_fingerprint() -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"fingerprint": build_fingerprint()}, fh, indent=2)


def artifact_status(horizon: int = 1) -> dict[str, bool]:
    return {
        "fixture": (ROOT / CFG.data.fixture.path / "rainfall.csv").exists(),
        "dataset": (ROOT / CFG.data.processed).exists(),
        "features": (ROOT / "data/processed" / f"features_h{horizon}.parquet").exists(),
        "model": (ROOT / CFG.outputs.models / f"risk_random_forest_h{horizon}.pkl").exists(),
        "summary": (ROOT / CFG.outputs.reports / f"summary_h{horizon}.json").exists(),
    }


def ensure_artifacts(horizon: int = 1, verbose: bool = True) -> dict[str, bool]:
    """Build anything missing OR stale. Safe to call repeatedly."""
    status = artifact_status(horizon)

    def log(msg: str) -> None:
        if verbose:
            print(f"[bootstrap] {msg}")

    # Stale artifacts are worse than missing ones: they load cleanly and
    # then explode somewhere far away. If the fingerprint moved, treat
    # everything derived as absent.
    want, have = build_fingerprint(), _stored_fingerprint()
    if have != want:
        if have is not None:
            log(f"config changed ({have} -> {want}) — rebuilding derived artifacts")
        # The FIXTURE is invalidated too. It is committed to the repo, so a
        # schema change (a renamed or added column) leaves a stale copy on
        # every deploy until someone remembers to regenerate and commit it
        # — which is exactly how the hosted app ended up rejecting its own
        # fixture for missing 'basin' and 'state'. Regenerating is cheap and
        # deterministic (fixed seed), so just always do it.
        for key in ("fixture", "dataset", "features", "model", "summary"):
            status[key] = False

    if not status["fixture"]:
        # The fixture is always generated, even when config asks for real
        # data: it is the fallback the loaders need if the real files turn
        # out to be absent (gitignored on a deploy, say).
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

    _write_fingerprint()
    final = artifact_status(horizon)
    log("ready" if all(final.values()) else f"incomplete: {final}")
    return final


if __name__ == "__main__":
    ensure_artifacts()
