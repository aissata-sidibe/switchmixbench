"""Small JSON / JSONL I/O helpers used across SwitchMixBench.

The benchmark deliberately keeps file formats simple and line‑oriented so that
datasets can be streamed and manipulated with standard tooling. This module
centralises common read/write helpers to avoid duplicating boilerplate.
"""

import json
from pathlib import Path


def read_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def read_any(path):
    """Load either JSON or JSONL depending on the filename extension.

    ``*.json`` files are parsed as a single JSON document; ``*.jsonl`` files
    are read as a list of JSON objects, one per line.
    """
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    return read_json(path)

def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
