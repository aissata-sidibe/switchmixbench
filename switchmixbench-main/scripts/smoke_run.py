"""Lightweight smoke test for SwitchMixBench installation.

This script:
  1) builds a tiny in-memory JSONL dataset in a temporary directory
  2) runs the tokenizer analysis module on that dataset using a small
     fake tokenizer (no network or model downloads)
  3) writes a CSV under results/tables/ and prints a short 'OK' message.

It is intended as a quick health check that runs in a few seconds on CPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from switchmixbench.analysis import tokenizer_analysis as ta
from switchmixbench.utils.io import write_jsonl


class _SmokeFakeTokenizer:
    """Minimal tokenizer used only for the smoke test.

    It mimics the subset of the transformers AutoTokenizer API that
    `compute_tokenizer_stats` relies on, but operates locally and does
    not perform any network calls or downloads.
    """

    def __call__(self, text, truncation=False, add_special_tokens=True):
        n = max(1, len(str(text).split()))
        return {"input_ids": list(range(n))}


def _build_toy_dataset(tmp: Path) -> Path:
    rows = [
        {
            "id": "ex1_clean",
            "pair_id": "ex1",
            "task": "nli",
            "split": "test",
            "variant": "clean",
            "input": "Le chat est sur le tapis.",
            "label": "entailment",
        },
        {
            "id": "ex1_pert",
            "pair_id": "ex1",
            "task": "nli",
            "split": "test",
            "variant": "perturbed",
            "input": "Le chat est on the rug.",
            "label": "entailment",
        },
    ]
    path = tmp / "smoke_nli.jsonl"
    write_jsonl(rows, path)
    return path


def main() -> None:
    # Use a temp dir so that the smoke test is non-invasive.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        data_path = _build_toy_dataset(tmp)

        # Patch the tokenizer factory so no real models are loaded.
        ta._lazy_tokenizer = lambda _name: _SmokeFakeTokenizer()  # type: ignore[attr-defined]

        out_csv = ta.run_tokenizer_analysis(
            data_paths=[str(data_path)],
            tokenizer_name_or_path="smoke-fake-tokenizer",
            out_csv="results/tables/tokenizer_stats_smoke.csv",
            max_pairs=10,
        )

        # Basic sanity check: CSV exists and has at least one row.
        import pandas as pd

        df = pd.read_csv(out_csv)
        assert len(df) >= 1
        assert np.all(df["n_pairs"].to_numpy(dtype=float) >= 1)

    print("SwitchMixBench smoke test: OK")


if __name__ == "__main__":
    main()

