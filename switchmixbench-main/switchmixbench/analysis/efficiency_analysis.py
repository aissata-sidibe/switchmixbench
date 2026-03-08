from __future__ import annotations

"""Efficiency impact analysis for SwitchMixBench.

This module estimates how much the perturbations used in SwitchMixBench affect
inference efficiency for encoder models. It measures sequence-length
inflation, latency differences, and a simple activation memory proxy between
clean and perturbed variants of the same underlying example.
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from switchmixbench.utils.io import read_any


def _lazy_torch_transformers():
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Efficiency analysis requires torch + transformers.") from e
    return torch, AutoTokenizer, AutoModel


def _pair_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Group dataset rows into clean / perturbed pairs."""
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        pid = r.get("pair_id")
        if pid is None:
            rid = str(r.get("id", ""))
            pid = rid.split("__")[0] if "__" in rid else rid
        pid = str(pid)
        var = str(r.get("variant", ""))
        pairs[pid][var] = r
    return pairs


def _hidden_size_from_config(cfg) -> int:
    """Infer hidden size from a model config object."""
    for key in ["hidden_size", "d_model", "dim"]:
        if hasattr(cfg, key):
            v = getattr(cfg, key)
            if isinstance(v, int):
                return v
    return 768


def _estimate_activation_bytes(seq_len: int, hidden_size: int, dtype_bytes: int = 4) -> int:
    """Rough activation memory proxy in bytes for one sequence."""
    return int(seq_len * hidden_size * dtype_bytes)


def run_efficiency_analysis(
    data_paths: List[str],
    model_name_or_path: str,
    out_csv: str = "results/tables/efficiency_metrics.csv",
    max_pairs: int = 200,
    max_length: int = 256,
    device: Optional[str] = None,
    include_tokenization_time: bool = False,
) -> str:
    """Measure efficiency differences between clean and perturbed variants."""
    torch, AutoTokenizer, AutoModel = _lazy_torch_transformers()

    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    hidden_size = _hidden_size_from_config(model.config)

    records: List[Dict[str, Any]] = []

    for path in data_paths:
        rows = read_any(path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected list rows in {path}")
        pairs = _pair_rows(rows)
        pair_ids = list(pairs.keys())
        if max_pairs is not None and max_pairs >= 0:
            pair_ids = pair_ids[: max_pairs]

        clean_lens = []
        pert_lens = []
        clean_lat_ms = []
        pert_lat_ms = []

        with torch.no_grad():
            for pid in pair_ids:
                d = pairs[pid]
                clean = d.get("clean")
                pert = d.get("perturbed") or d.get("switchmix")
                if clean is None or pert is None:
                    continue

                clean_text = str(clean.get("input") or clean.get("prompt") or "")
                pert_text = str(pert.get("input") or pert.get("prompt") or "")

                # Tokenization
                t_tok0 = time.perf_counter()
                enc_c = tok(clean_text, return_tensors="pt", truncation=True, max_length=max_length)
                enc_p = tok(pert_text, return_tensors="pt", truncation=True, max_length=max_length)
                t_tok1 = time.perf_counter()

                enc_c = {k: v.to(device) for k, v in enc_c.items()}
                enc_p = {k: v.to(device) for k, v in enc_p.items()}

                c_len = int(enc_c["input_ids"].shape[1])
                p_len = int(enc_p["input_ids"].shape[1])
                clean_lens.append(c_len)
                pert_lens.append(p_len)

                # Forward latency
                t0 = time.perf_counter()
                _ = model(**enc_c)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                _ = model(**enc_p)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t3 = time.perf_counter()

                if include_tokenization_time:
                    tok_ms_each = ((t_tok1 - t_tok0) * 1000.0) / 2.0
                    clean_lat_ms.append((t1 - t0) * 1000.0 + tok_ms_each)
                    pert_lat_ms.append((t3 - t2) * 1000.0 + tok_ms_each)
                else:
                    clean_lat_ms.append((t1 - t0) * 1000.0)
                    pert_lat_ms.append((t3 - t2) * 1000.0)

        task = str(rows[0].get("task", "")) if rows else ""
        split = str(rows[0].get("split", "")) if rows else ""

        avg_c_len = float(np.mean(clean_lens)) if clean_lens else 0.0
        avg_p_len = float(np.mean(pert_lens)) if pert_lens else 0.0

        records.append(
            {
                "data_path": str(path),
                "model": model_name_or_path,
                "device": device,
                "max_length": int(max_length),
                "n_pairs": int(len(clean_lens)),
                "avg_seq_len_clean": avg_c_len,
                "avg_seq_len_perturbed": avg_p_len,
                "avg_seq_len_delta": avg_p_len - avg_c_len,
                "avg_latency_ms_clean": float(np.mean(clean_lat_ms)) if clean_lat_ms else 0.0,
                "avg_latency_ms_perturbed": float(np.mean(pert_lat_ms)) if pert_lat_ms else 0.0,
                "avg_latency_ms_delta": (float(np.mean(pert_lat_ms)) - float(np.mean(clean_lat_ms)))
                if clean_lat_ms and pert_lat_ms
                else 0.0,
                "hidden_size": int(hidden_size),
                "activation_bytes_clean_est": _estimate_activation_bytes(int(round(avg_c_len)), hidden_size),
                "activation_bytes_perturbed_est": _estimate_activation_bytes(int(round(avg_p_len)), hidden_size),
                "activation_bytes_delta_est": _estimate_activation_bytes(int(round(avg_p_len)), hidden_size)
                - _estimate_activation_bytes(int(round(avg_c_len)), hidden_size),
                "include_tokenization_time": bool(include_tokenization_time),
            }
        )

    df = pd.DataFrame.from_records(records)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)

