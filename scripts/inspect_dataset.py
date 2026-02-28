from __future__ import annotations

import sys
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from switchmixbench.utils.io import read_any


def _len_stats(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p: float) -> float:
        i = int(round((n - 1) * p))
        return float(xs_sorted[max(0, min(n - 1, i))])

    return {
        "mean": float(sum(xs_sorted) / n),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": float(xs_sorted[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to JSON or JSONL dataset file")
    ap.add_argument("--show", type=int, default=3, help="Number of random pairs to print")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_any(args.path)
    if not isinstance(rows, list):
        raise ValueError("Expected dataset to be a list of rows.")

    print("path:", args.path)
    print("n_rows:", len(rows))

    by_variant = Counter([r.get("variant", "") for r in rows])
    by_task = Counter([r.get("task", "") for r in rows])
    by_split = Counter([r.get("split", "") for r in rows])

    print("by_task:", dict(by_task))
    print("by_split:", dict(by_split))
    print("by_variant:", dict(by_variant))

    # Pairing sanity check
    pairs = defaultdict(list)
    for r in rows:
        pid = r.get("pair_id") or (str(r.get("id", "")).split("__")[0] if r.get("id") else None)
        if pid is not None:
            pairs[str(pid)].append(r)

    sizes = Counter([len(v) for v in pairs.values()])
    print("pair_size_histogram:", dict(sizes))

    # Length stats (character length of input)
    clean_lens, pert_lens = [], []
    for v in pairs.values():
        clean = next((x for x in v if x.get("variant") == "clean"), None)
        pert = next((x for x in v if x.get("variant") in ["perturbed", "switchmix"]), None)
        if clean is not None:
            clean_lens.append(len(str(clean.get("input") or clean.get("prompt") or "")))
        if pert is not None:
            pert_lens.append(len(str(pert.get("input") or pert.get("prompt") or "")))

    print("clean_input_len_stats:", _len_stats(clean_lens))
    print("perturbed_input_len_stats:", _len_stats(pert_lens))

    rng = random.Random(args.seed)
    pair_ids = list(pairs.keys())
    rng.shuffle(pair_ids)
    to_show = pair_ids[: max(0, args.show)]

    for pid in to_show:
        v = pairs[pid]
        clean = next((x for x in v if x.get("variant") == "clean"), None)
        pert = next((x for x in v if x.get("variant") in ["perturbed", "switchmix"]), None)
        print("\n--- pair_id:", pid, "---")
        if clean:
            print("[clean] label:", clean.get("label"))
            print((clean.get("input") or clean.get("prompt") or "")[:400])
        if pert:
            print("[perturbed] label:", pert.get("label"))
            print((pert.get("input") or pert.get("prompt") or "")[:400])


if __name__ == "__main__":
    main()

