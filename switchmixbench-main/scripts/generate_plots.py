"""Generate publication-ready plots from SwitchMixBench results tables.

This script creates:

1. Robustness scaling plot
   - Input:  results/tables/scaling_results_large.csv
   - Output: results/figures/robustness_scaling.png

2. Cross-language robustness plot
   - Input:  results/tables/cross_language_robustness.csv
   - Output: results/figures/cross_language_robustness.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_figures_dir() -> Path:
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _plot_robustness_scaling(fig_dir: Path) -> None:
    path = Path("results/tables/scaling_results_large.csv")
    if not path.exists():
        print(f"[generate_plots] Skipping robustness scaling plot; missing {path}.")
        return

    df = pd.read_csv(path)
    if "estimated_model_size" not in df.columns or "robustness_drop" not in df.columns:
        print("[generate_plots] scaling_results_large.csv missing required columns.")
        return

    plt.figure(figsize=(6, 4))
    # Scatter plot, coloured / labelled by model.
    for model_name, group in df.groupby("model"):
        plt.scatter(
            group["estimated_model_size"],
            group["robustness_drop"],
            label=model_name,
        )

    plt.xlabel("Estimated model size (M parameters)")
    plt.ylabel("Robustness drop (clean − switchmix)")
    plt.title("Robustness vs model size (FR–EN)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.2)

    out_path = fig_dir / "robustness_scaling.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


def _plot_cross_language(fig_dir: Path) -> None:
    path = Path("results/tables/cross_language_robustness.csv")
    if not path.exists():
        print(f"[generate_plots] Skipping cross-language plot; missing {path}.")
        return

    df = pd.read_csv(path)
    required = {"language_pair", "model", "robustness_drop"}
    if not required.issubset(df.columns):
        print("[generate_plots] cross_language_robustness.csv missing required columns.")
        return

    plt.figure(figsize=(6, 4))
    language_pairs = sorted(df["language_pair"].unique())
    models = sorted(df["model"].unique())

    x_positions = range(len(language_pairs))
    width = 0.8 / max(1, len(models))

    for i, model in enumerate(models):
        subset = df[df["model"] == model]
        # Align bars with the language_pair order
        heights = [
            float(
                subset[subset["language_pair"] == lp]["robustness_drop"].mean()
                if not subset[subset["language_pair"] == lp].empty
                else 0.0
            )
            for lp in language_pairs
        ]
        offsets = [x + (i - (len(models) - 1) / 2.0) * width for x in x_positions]
        plt.bar(offsets, heights, width=width, label=model)

    plt.xticks(list(x_positions), language_pairs)
    plt.xlabel("Language pair")
    plt.ylabel("Robustness drop (clean − switchmix)")
    plt.title("Cross-language robustness")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, axis="y", alpha=0.2)

    out_path = fig_dir / "cross_language_robustness.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


def main() -> None:
    fig_dir = _ensure_figures_dir()
    _plot_robustness_scaling(fig_dir)
    _plot_cross_language(fig_dir)


if __name__ == "__main__":
    main()

