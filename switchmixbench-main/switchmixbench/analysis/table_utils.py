"""Helpers for generating paper-ready tables from SwitchMixBench results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_paper_tables() -> str:
    """Create a consolidated CSV table for paper results.

    The function reads:

    - `results/tables/cross_language_robustness.csv`

    and writes:

    - `results/tables/paper_results_table.csv`

    with columns:

    - model
    - language_pair
    - clean_accuracy
    - switchmix_accuracy
    - robustness_drop

    Returns
    -------
    str
        Path to the written CSV.
    """

    tables_dir = Path("results/tables")
    in_path = tables_dir / "cross_language_robustness.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing cross-language results: {in_path}. "
            "Run `scripts/run_cross_language_analysis.py` first."
        )

    df = pd.read_csv(in_path)
    cols = [
        "model",
        "language_pair",
        "clean_accuracy",
        "switchmix_accuracy",
        "robustness_drop",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"cross_language_robustness.csv is missing required columns: {missing}"
        )

    out_df = df[cols].copy()
    out_path = tables_dir / "paper_results_table.csv"
    out_df.to_csv(out_path, index=False)
    return str(out_path)

