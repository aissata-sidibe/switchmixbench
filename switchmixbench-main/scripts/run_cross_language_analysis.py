"""Aggregate robustness summaries across language pairs.

This script reads one or more `robustness_summary*.csv` tables under
`results/tables/` and produces a unified cross-language view:

    results/tables/cross_language_robustness.csv

It does not modify any existing CSV formats.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List

import pandas as pd


def _infer_language_pair_from_path(path: Path) -> str:
    name = path.name
    if name == "robustness_summary.csv":
        # Default benchmark configuration is FR–EN.
        return "fr-en"
    if name.startswith("robustness_summary_") and name.endswith(".csv"):
        # e.g. robustness_summary_fr-es.csv -> fr-es
        return name[len("robustness_summary_") : -4]
    # Fallback: use the stem as-is.
    return name[:-4]


def main() -> None:
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        Path(p) for p in glob.glob(str(tables_dir / "robustness_summary*.csv"))
    )
    if not paths:
        print("No robustness_summary CSV files found under results/tables/.")
        return

    rows: List[dict] = []
    for path in paths:
        lang_pair = _infer_language_pair_from_path(path)
        df = pd.read_csv(path)
        # We expect columns: model, clean_score, switchmix_score, robustness_drop
        for _, r in df.iterrows():
            rows.append(
                {
                    "language_pair": lang_pair,
                    "model": r.get("model"),
                    "clean_accuracy": r.get("clean_score"),
                    "switchmix_accuracy": r.get("switchmix_score"),
                    "robustness_drop": r.get("robustness_drop"),
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = tables_dir / "cross_language_robustness.csv"
    out_df.to_csv(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    main()

