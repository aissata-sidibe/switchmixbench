from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from switchmixbench.utils.io import read_any


def _lazy_transformers():
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Representation analysis requires torch + transformers."
        ) from e
    return torch, AutoTokenizer, AutoModel


def _pair_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
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


def _pool(hidden, attention_mask, pool: str):
    # hidden: [B, T, H]
    if pool == "cls":
        return hidden[:, 0, :]
    if pool == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B,T,1]
        s = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return s / denom
    raise ValueError(f"Unknown pool method: {pool}")


def compute_representation_shift(
    data_paths: List[str],
    model_name_or_path: str,
    pool: str = "cls",
    max_pairs: Optional[int] = None,
    max_length: int = 256,
    device: Optional[str] = None,
) -> pd.DataFrame:
    torch, AutoTokenizer, AutoModel = _lazy_transformers()

    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    records: List[Dict[str, Any]] = []

    for path in data_paths:
        rows = read_any(path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected list rows in {path}")

        pairs = _pair_rows(rows)
        pair_ids = list(pairs.keys())
        if max_pairs is not None:
            pair_ids = pair_ids[: max_pairs]

        # Accumulators: (task, split, layer) -> list[cos]
        cos_by_key: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)

        with torch.no_grad():
            for pid in pair_ids:
                d = pairs[pid]
                clean = d.get("clean")
                pert = d.get("perturbed") or d.get("switchmix")
                if clean is None or pert is None:
                    continue

                task = str(clean.get("task", pert.get("task", "")))
                split = str(clean.get("split", pert.get("split", "")))

                clean_text = str(clean.get("input") or clean.get("prompt") or "")
                pert_text = str(pert.get("input") or pert.get("prompt") or "")

                enc_c = tok(
                    clean_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                enc_p = tok(
                    pert_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )

                enc_c = {k: v.to(device) for k, v in enc_c.items()}
                enc_p = {k: v.to(device) for k, v in enc_p.items()}

                out_c = model(**enc_c)
                out_p = model(**enc_p)

                hs_c = out_c.hidden_states  # tuple(layer+emb)
                hs_p = out_p.hidden_states
                if hs_c is None or hs_p is None:
                    continue

                n_layers = min(len(hs_c), len(hs_p))
                for li in range(n_layers):
                    v_c = _pool(hs_c[li], enc_c.get("attention_mask"), pool=pool)  # [1,H]
                    v_p = _pool(hs_p[li], enc_p.get("attention_mask"), pool=pool)
                    v_c = v_c[0]
                    v_p = v_p[0]

                    cos = torch.nn.functional.cosine_similarity(v_c, v_p, dim=0).item()
                    cos_by_key[(task, split, li)].append(float(cos))

        for (task, split, layer), vals in sorted(cos_by_key.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            arr = np.array(vals, dtype=np.float64)
            records.append(
                {
                    "data_path": str(path),
                    "model": model_name_or_path,
                    "pool": pool,
                    "task": task,
                    "split": split,
                    "layer": int(layer),
                    "n_pairs": int(len(vals)),
                    "mean_cosine": float(arr.mean()) if len(arr) else 0.0,
                    "std_cosine": float(arr.std(ddof=0)) if len(arr) else 0.0,
                    "mean_drift": float((1.0 - arr).mean()) if len(arr) else 0.0,
                }
            )

    return pd.DataFrame.from_records(records)


def run_representation_analysis(
    data_paths: List[str],
    model_name_or_path: str,
    out_csv: str = "results/tables/representation_shift.csv",
    pool: str = "cls",
    max_pairs: Optional[int] = None,
    max_length: int = 256,
    device: Optional[str] = None,
) -> str:
    df = compute_representation_shift(
        data_paths=data_paths,
        model_name_or_path=model_name_or_path,
        pool=pool,
        max_pairs=max_pairs,
        max_length=max_length,
        device=device,
    )
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)

