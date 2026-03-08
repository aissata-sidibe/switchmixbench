from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from switchmixbench.analysis.representation_analysis import run_representation_analysis
from switchmixbench.utils.config import get, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_paths = get(cfg, "data_paths", [])
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    model = str(get(cfg, "model", "xlm-roberta-base"))
    out_csv = str(get(cfg, "output_csv", "results/tables/representation_shift.csv"))
    pool = str(get(cfg, "pool", "cls"))
    max_pairs = get(cfg, "max_pairs", 200)
    max_length = int(get(cfg, "max_length", 256))
    device = get(cfg, "device", None)

    out = run_representation_analysis(
        data_paths=list(data_paths),
        model_name_or_path=model,
        out_csv=out_csv,
        pool=pool,
        max_pairs=max_pairs,
        max_length=max_length,
        device=device,
    )
    print("saved:", out)


if __name__ == "__main__":
    main()

