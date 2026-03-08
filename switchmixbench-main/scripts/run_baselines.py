import sys
from pathlib import Path

import argparse
import random
import pandas as pd

# Allow running as: `python scripts/run_baselines.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from switchmixbench.utils.config import get, load_yaml
from switchmixbench.utils.io import read_any

NLI_LABELS = ["entailment", "contradiction", "neutral"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/switchmixbench.json")
    p.add_argument("--task", type=str, choices=["nli", "qa"], required=True)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config; when provided, can specify 'data' and 'task'.",
    )
    args = p.parse_args()

    if args.config is not None:
        cfg = load_yaml(args.config)
        data_path = get(cfg, "data", args.data)
        task = get(cfg, "task", args.task)
    else:
        data_path = args.data
        task = args.task

    data = read_any(data_path)
    rows = [r for r in data if r["task"] == task]

    out = []
    for r in rows:
        if task == "nli":
            pred = random.choice(NLI_LABELS)
        else:
            pred = ""
        out.append({**r, "prediction": pred, "model": "baseline_random"})

    df = pd.DataFrame(out)
    path = f"results/tables/{task}_baseline_random.csv"
    df.to_csv(path, index=False)
    print("saved:", path)

if __name__ == "__main__":
    main()
