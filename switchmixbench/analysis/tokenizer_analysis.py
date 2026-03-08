from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from switchmixbench.utils.io import read_any


def _lazy_tokenizer(model_name_or_path: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Tokenizer analysis requires transformers. Install with: pip install transformers"
        ) from e
    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence in bits.
    p, q must be non-negative and sum to 1.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _chars_no_ws(s: str) -> int:
    return sum(1 for c in s if not c.isspace())


def _word_count(s: str) -> int:
    s = s.strip()
    if not s:
        return 0
    return len(s.split())


def _pair_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns: pair_id -> variant -> row
    Accepts variant names:
      clean, perturbed (preferred)
      clean, switchmix (legacy)
    """
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


def compute_tokenizer_stats(
    data_paths: List[str],
    tokenizer_name_or_path: str,
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    tok = _lazy_tokenizer(tokenizer_name_or_path)

    records: List[Dict[str, Any]] = []

    for path in data_paths:
        rows = read_any(path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected list rows in {path}")

        pairs = _pair_rows(rows)
        pair_ids = list(pairs.keys())
        if max_pairs is not None:
            pair_ids = pair_ids[: max_pairs]

        # Aggregation buckets by (task, split)
        buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        tok_counts_clean: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        tok_counts_pert: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

        for pid in pair_ids:
            d = pairs[pid]
            clean = d.get("clean")
            pert = d.get("perturbed") or d.get("switchmix")
            if clean is None or pert is None:
                continue

            task = str(clean.get("task", pert.get("task", "")))
            split = str(clean.get("split", pert.get("split", "")))
            key = (task, split)
            b = buckets.setdefault(
                key,
                {
                    "n_pairs": 0,
                    "clean_tokens": [],
                    "pert_tokens": [],
                    "clean_frag": [],
                    "pert_frag": [],
                    "clean_tpc": [],
                    "pert_tpc": [],
                    "inflation": [],
                    "increase": [],
                },
            )

            clean_text = str(clean.get("input") or clean.get("prompt") or "")
            pert_text = str(pert.get("input") or pert.get("prompt") or "")

            clean_ids = tok(clean_text, truncation=False, add_special_tokens=True)["input_ids"]
            pert_ids = tok(pert_text, truncation=False, add_special_tokens=True)["input_ids"]

            c_len = int(len(clean_ids))
            p_len = int(len(pert_ids))

            c_words = max(1, _word_count(clean_text))
            p_words = max(1, _word_count(pert_text))
            c_chars = max(1, _chars_no_ws(clean_text))
            p_chars = max(1, _chars_no_ws(pert_text))

            b["n_pairs"] += 1
            b["clean_tokens"].append(c_len)
            b["pert_tokens"].append(p_len)
            b["increase"].append(p_len - c_len)
            b["inflation"].append(p_len / max(1, c_len))
            b["clean_frag"].append(c_len / c_words)
            b["pert_frag"].append(p_len / p_words)
            b["clean_tpc"].append(c_len / c_chars)
            b["pert_tpc"].append(p_len / p_chars)

            tok_counts_clean[key].update(clean_ids)
            tok_counts_pert[key].update(pert_ids)

        for (task, split), b in buckets.items():
            if b["n_pairs"] == 0:
                continue

            # Token distribution shift
            vocab = set(tok_counts_clean[(task, split)].keys()) | set(tok_counts_pert[(task, split)].keys())
            vocab = list(vocab)
            pc = np.array([tok_counts_clean[(task, split)].get(t, 0) for t in vocab], dtype=np.float64)
            pp = np.array([tok_counts_pert[(task, split)].get(t, 0) for t in vocab], dtype=np.float64)
            js = _js_divergence(pc / max(1.0, pc.sum()), pp / max(1.0, pp.sum()))

            records.append(
                {
                    "data_path": str(path),
                    "task": task,
                    "split": split,
                    "tokenizer": tokenizer_name_or_path,
                    "n_pairs": int(b["n_pairs"]),
                    "avg_clean_tokens": float(np.mean(b["clean_tokens"])),
                    "avg_perturbed_tokens": float(np.mean(b["pert_tokens"])),
                    "avg_token_increase": float(np.mean(b["increase"])),
                    "avg_length_inflation": float(np.mean(b["inflation"])),
                    "avg_fragmentation_clean": float(np.mean(b["clean_frag"])),
                    "avg_fragmentation_perturbed": float(np.mean(b["pert_frag"])),
                    "avg_tokens_per_char_clean": float(np.mean(b["clean_tpc"])),
                    "avg_tokens_per_char_perturbed": float(np.mean(b["pert_tpc"])),
                    "token_js_divergence_bits": float(js),
                }
            )

    return pd.DataFrame.from_records(records)


def run_tokenizer_analysis(
    data_paths: List[str],
    tokenizer_name_or_path: str,
    out_csv: str = "results/tables/tokenizer_stats.csv",
    max_pairs: Optional[int] = None,
) -> str:
    df = compute_tokenizer_stats(
        data_paths=data_paths,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_pairs=max_pairs,
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)

