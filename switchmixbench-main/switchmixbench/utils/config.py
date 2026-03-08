from __future__ import annotations

"""YAML-based configuration helpers for SwitchMixBench.

The benchmark relies on small, human-readable YAML files to configure dataset
building and analysis scripts. This module contains a minimal loader and a
few convenience utilities for working with nested configuration dictionaries.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _lazy_import_yaml():
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "YAML support requires pyyaml. Install with: pip install pyyaml"
        ) from e
    return yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    yaml = _lazy_import_yaml()
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a YAML mapping at {path}, got: {type(obj)}")
    return obj


def get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Dot-path accessor for nested mapping-like configurations.

    Examples
    --------
    ``get(cfg, \"source.dataset_name\")`` reads ``cfg[\"source\"][\"dataset_name\"]``
    when present, falling back to ``default`` otherwise.
    """
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class ExperimentPaths:
    data_path: Path
    results_dir: Path = Path("results")
    tables_dir: Path = Path("results/tables")

    def ensure(self) -> "ExperimentPaths":
        """Create result directories on disk if they do not exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        return self

