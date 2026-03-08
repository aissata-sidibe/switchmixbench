from __future__ import annotations

"""CLI entrypoint for multi-model scaling robustness experiments."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from switchmixbench.analysis.scaling_analysis import run_scaling_experiment
from switchmixbench.utils.config import get, load_yaml


DEFAULT_MODELS = [
    "distilbert-base-multilingual-cased",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "xlm-roberta-large",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Allow both "test_paths" (new) and "eval_paths" (legacy/fast config).
    train_paths = get(cfg, "train_paths", [])
    test_paths = get(cfg, "test_paths", get(cfg, "eval_paths", []))
    if isinstance(train_paths, str):
        train_paths = [train_paths]
    if isinstance(test_paths, str):
        test_paths = [test_paths]

    raw_models = get(cfg, "models", DEFAULT_MODELS)
    # Support several shapes:
    # - list of model names (recommended)
    # - single string
    # - mapping of groups -> [models] (as in scaling_analysis_fast.yaml)
    if isinstance(raw_models, str):
        model_names = [raw_models]
    elif isinstance(raw_models, dict):
        model_names = []
        for grp in raw_models.values():
            if isinstance(grp, str):
                model_names.append(grp)
            elif isinstance(grp, (list, tuple)):
                model_names.extend(list(grp))
    else:
        model_names = list(raw_models)
    out_csv = str(get(cfg, "output_csv", "results/tables/scaling_results.csv"))

    out = run_scaling_experiment(
        train_paths=list(train_paths),
        test_paths=list(test_paths),
        model_names=list(model_names),
        out_csv=out_csv,
        seed=int(get(cfg, "seed", 42)),
        max_length=int(get(cfg, "max_length", get(cfg, "max_length", 256))),
        train_split=str(get(cfg, "train_split", "train")),
        test_split=str(get(cfg, "test_split", "test")),
        train_variant=str(get(cfg, "train_variant", "clean")),
        test_clean_variant=str(get(cfg, "test_clean_variant", "clean")),
        test_pert_variant=str(get(cfg, "test_pert_variant", "perturbed")),
        max_train_examples=get(cfg, "max_train_examples", None),
        max_test_examples=get(cfg, "max_test_examples", None),
        epochs=float(get(cfg, "epochs", get(cfg, "epochs", 1.0))),
        batch_size=int(get(cfg, "batch_size", get(cfg, "per_device_train_batch_size", 16))),
        lr=float(get(cfg, "lr", 2e-5)),
    )
    print("saved:", out)


if __name__ == "__main__":
    main()

