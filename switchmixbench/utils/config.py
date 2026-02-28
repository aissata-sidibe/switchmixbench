from __future__ import annotations

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
    """
    Dot-path accessor for nested dict configs.
    Example: get(cfg, "source.dataset_name")
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
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        return self

