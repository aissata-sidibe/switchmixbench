from pathlib import Path

import numpy as np
import torch

from switchmixbench.analysis import representation_analysis as ra
from switchmixbench.utils.io import write_jsonl


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=8):
        n = min(max_length, max(1, len(str(text).split())))
        ids = torch.arange(n, dtype=torch.long).unsqueeze(0)
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn}


class _FakeOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeModel(torch.nn.Module):
    def __init__(self, n_layers: int = 3, hidden_size: int = 8):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

    def forward(self, **kwargs):
        # Build a small stack of identical hidden states [layer, batch, seq, dim]
        input_ids = kwargs["input_ids"]
        b, t = input_ids.shape
        hs = []
        base = torch.ones(b, t, self.hidden_size, dtype=torch.float32)
        for _ in range(self.n_layers):
            hs.append(base)
        return _FakeOutput(tuple(hs))


def _fake_lazy_transformers():
    """Return fake torch + transformers objects with from_pretrained() methods."""

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
    path = tmp_path / "toy_rep.jsonl"
    write_jsonl(rows, path)
    return str(path)


def test_representation_analysis_schema_and_bounds(monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "_lazy_transformers", _fake_lazy_transformers)

    data_path = _make_small_dataset(tmp_path)
    df = ra.compute_representation_shift(
        data_paths=[data_path],
        model_name_or_path="dummy-model",
        pool="cls",
        max_pairs=1,
        max_length=8,
        device="cpu",
    )

    assert len(df) >= 1

    required = {
        "data_path",
        "model",
        "task",
        "split",
        "layer",
        "n_pairs",
        "mean_cosine",
        "mean_drift",
    }
    assert required.issubset(df.columns)

    cos = df["mean_cosine"].to_numpy(dtype=float)
    drift = df["mean_drift"].to_numpy(dtype=float)

    # Cosine similarity must be within [-1, 1], drift must be >= 0.
    assert np.all(cos >= -1.0) and np.all(cos <= 1.0)
    assert np.all(drift >= 0.0)

