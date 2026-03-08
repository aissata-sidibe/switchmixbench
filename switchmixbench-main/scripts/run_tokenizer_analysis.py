from __future__ import annotations

"""CLI entrypoint for tokenizer-level robustness analysis.

This script reads a YAML config describing input dataset paths and the
tokeniser to use, then calls `switchmixbench.analysis.tokenizer_analysis`
and writes a CSV table under `results/tables/`.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from switchmixbench.analysis.tokenizer_analysis import run_tokenizer_analysis
from switchmixbench.utils.config import get, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_paths = get(cfg, "data_paths", [])
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    tokenizer = str(get(cfg, "tokenizer", "xlm-roberta-base"))
    out_csv = str(get(cfg, "output_csv", "results/tables/tokenizer_stats.csv"))
    max_pairs = get(cfg, "max_pairs", None)

    out = run_tokenizer_analysis(
        data_paths=list(data_paths),
        tokenizer_name_or_path=tokenizer,
        out_csv=out_csv,
        max_pairs=max_pairs,
    )
    print("saved:", out)


if __name__ == "__main__":
    main()

