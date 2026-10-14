"""Configuration loading. Everything in the pipeline reads from config.yaml."""

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """Dict with attribute access and nested-key lookup."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def resolve(self, relative) -> Path:
        """Resolve a config-declared relative path against the repo root.

        Deliberately not named `path` — config keys are exposed as
        attributes, and `data.fixture.path` is one of them.
        """
        return ROOT / relative


def load(config_path: str | Path | None = None) -> Config:
    path = Path(config_path) if config_path else ROOT / "config.yaml"
    with open(path) as fh:
        return Config(yaml.safe_load(fh))


CFG = load()
