from pathlib import Path

import numpy as np

from switchmixbench.analysis import tokenizer_analysis as ta
from switchmixbench.utils.io import write_jsonl


class FakeTokenizer:
    """Very small stand-in for Hugging Face tokenizers."""

    def __call__(self, text, truncation=False, add_special_tokens=True):
        # Simple deterministic mapping: one token id per whitespace-delimited token.
        n = max(1, len(str(text).split()))
        return {"input_ids": list(range(n))}


def _make_small_dataset(tmp_path: Path) -> str:
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
    path = tmp_path / "toy.jsonl"
    write_jsonl(rows, path)
    return str(path)


def test_tokenizer_analysis_schema_and_non_negative(monkeypatch, tmp_path):
    # Avoid downloading real models by patching the lazy tokenizer factory.
    monkeypatch.setattr(ta, "_lazy_tokenizer", lambda name: FakeTokenizer())

    data_path = _make_small_dataset(tmp_path)
    df = ta.compute_tokenizer_stats([data_path], tokenizer_name_or_path="dummy-tokenizer", max_pairs=10)

    # Expected columns
    expected_cols = {
        "data_path",
        "task",
        "split",
        "tokenizer",
        "n_pairs",
        "avg_clean_tokens",
        "avg_perturbed_tokens",
        "avg_token_increase",
        "avg_length_inflation",
        "avg_fragmentation_clean",
        "avg_fragmentation_perturbed",
        "avg_tokens_per_char_clean",
        "avg_tokens_per_char_perturbed",
        "token_js_divergence_bits",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) >= 1

    # All numeric metrics are finite and non-negative
    numeric_cols = [
        "n_pairs",
        "avg_clean_tokens",
        "avg_perturbed_tokens",
        "avg_token_increase",
        "avg_length_inflation",
        "avg_fragmentation_clean",
        "avg_fragmentation_perturbed",
        "avg_tokens_per_char_clean",
        "avg_tokens_per_char_perturbed",
        "token_js_divergence_bits",
    ]
    for col in numeric_cols:
        vals = df[col].to_numpy(dtype=float)
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 0.0)

