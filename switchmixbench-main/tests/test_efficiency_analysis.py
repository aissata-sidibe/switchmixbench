from pathlib import Path

import torch

from switchmixbench.analysis import efficiency_analysis as ea
from switchmixbench.utils.io import write_jsonl


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=16):
        # Single example with deterministic length based on number of tokens (capped).
        n = min(max_length, max(1, len(str(text).split())))
        ids = torch.arange(n, dtype=torch.long).unsqueeze(0)
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn}


class _FakeConfig:
    hidden_size = 8


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()

    def forward(self, **kwargs):
        # Output is unused in the metrics, so keep it minimal.
        return torch.zeros(1)


def _fake_lazy_torch_transformers():
    """Return (torch, AutoTokenizer, AutoModel) stand-ins with from_pretrained()."""

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeTokenizer()

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeModel()

    return torch, _FakeAutoTokenizer, _FakeAutoModel


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
    path = tmp_path / "toy_eff.jsonl"
    write_jsonl(rows, path)
    return str(path)


def test_efficiency_analysis_schema_and_length_delta(monkeypatch, tmp_path):
    # Patch transformer loading to avoid any remote downloads.
    monkeypatch.setattr(ea, "_lazy_torch_transformers", _fake_lazy_torch_transformers)

    data_path = _make_small_dataset(tmp_path)
    out_csv = ea.run_efficiency_analysis(
        data_paths=[data_path],
        model_name_or_path="dummy-model",
        out_csv=str(tmp_path / "eff.csv"),
        max_pairs=1,
        max_length=16,
        device="cpu",
    )

    import pandas as pd

    df = pd.read_csv(out_csv)
    assert len(df) == 1

    required = {
        "data_path",
        "model",
        "avg_seq_len_clean",
        "avg_seq_len_perturbed",
        "avg_seq_len_delta",
    }
    assert required.issubset(df.columns)

    # Sequence length delta must be >= 0 (perturbed is at least as long as clean).
    assert df["avg_seq_len_delta"].iloc[0] >= 0.0

