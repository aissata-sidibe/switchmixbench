## Project Overview

SwitchMixBench is a small, focused research benchmark for evaluating the **robustness of multilingual foundation models** under:

- code-switching between French and English
- informal / noisy user text
- small controlled perturbations

The benchmark is designed to measure **brittleness**, not leaderboard performance: how quickly performance degrades when moving from clean, well‑formed text to realistic mixed-language inputs.

### High-level architecture

- **Data generation**
  - Seed or Hugging Face datasets (e.g. `xnli`) provide parallel French / English text.
  - Deterministic rules inject code-switching and informal noise to build paired `clean` vs `perturbed` examples.
  - Processed data are stored as JSONL under `data/processed/`.
- **Evaluation**
  - Baseline classifiers and generative models consume the processed data and log predictions to `results/tables/`.
- **Analysis**
  - Dedicated modules quantify robustness at multiple levels:
    - tokenizer behaviour
    - hidden-state representation stability
    - multi-model scaling trends
    - efficiency / latency impact

The codebase aims to be **small, readable, and reproducible**, suitable for academic review and public release.

---

## Installation

- **Python**: 3.11 or 3.12 is recommended (3.9+ is supported).
- **OS**: Linux, macOS, and Windows are supported; the examples below use Windows PowerShell paths.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install SwitchMixBench in editable mode

From the project root (the directory containing `pyproject.toml`):

```powershell
pip install -e .
```

This makes the `switchmixbench` package importable while you edit the code.

---

## How to Run the Benchmark

All commands below assume you are in the project root (`switchmixbench-main/`) with the virtual environment activated.

### 1. Build dataset

Build a French–English NLI dataset using XNLI (with synthetic fallback if HF download is unavailable):

```powershell
python scripts/build_dataset.py --config configs/dataset_xnli_fr_en.yaml
```

This generates, for example:

- `data/processed/nli_train_fr-en.jsonl`
- `data/processed/nli_test_fr-en.jsonl`

Inspect dataset stats and paired examples:

```powershell
python scripts/inspect_dataset.py --path data/processed/nli_test_fr-en.jsonl --show 3
```

### 2. Run baseline evaluation

The random-label baseline operates directly on the processed JSON/JSONL file:

```powershell
python scripts/run_baselines.py --data data/processed/switchmixbench.json --task nli
```

This writes a CSV containing predictions and labels:

- `results/tables/nli_baseline_random.csv`

For generative baselines (e.g. FLAN‑T5), you can also use:

```powershell
python switchmixbench/eval/run_eval.py --model google/flan-t5-small --task nli --split test
```

### 3. Run tokenizer analysis

```powershell
python scripts/run_tokenizer_analysis.py --config configs/tokenizer_analysis.yaml
```

### 4. Run representation analysis

```powershell
python scripts/run_representation_analysis.py --config configs/representation_analysis.yaml
```

### 5. Run scaling analysis

```powershell
python scripts/run_scaling_analysis.py --config configs/scaling_analysis.yaml
```

### 6. Run efficiency analysis

```powershell
python scripts/run_efficiency_analysis.py --config configs/efficiency_analysis.yaml
```

---

## Output Structure

- **`data/processed/`**
  - Contains JSON and JSONL files used by the benchmark.
  - JSONL rows follow a common schema, e.g.:
    - `task`: `nli`, `qa`, ...
    - `split`: `train`, `test`
    - `variant`: `clean`, `perturbed` (or `switchmix` for legacy)
    - `input` / `prompt`: model input text
    - `label` / `target`: ground-truth label or answer
    - `pair_id`: identifier linking clean vs perturbed variants
    - `metadata`: details of the applied perturbations and language pair.

- **`results/tables/`**
  - All analysis scripts write tidy CSV tables here:
    - `tokenizer_stats.csv`: token counts, fragmentation, JS divergence.
    - `representation_shift.csv`: layer-wise cosine similarity and drift.
    - `scaling_results.csv`: accuracy on clean vs perturbed by model size, plus robustness gap Δ.
    - `efficiency_metrics.csv`: sequence lengths, latency deltas, and activation-size estimates.
    - Additional per-model evaluation outputs, e.g.:
      - `nli_test_google_flan-t5-small.csv`
      - `nli_baseline_random.csv`

These files are designed for direct use in plotting notebooks or downstream statistical analysis.

---

## Reproducibility Notes

- **Random seeds**
  - Dataset generation uses explicit seeds in `configs/dataset_*.yaml`.
  - Scaling experiments set a global seed via `transformers.set_seed`.
  - Perturbation functions in `switchmixbench.generate.*` take explicit `seed` arguments and are tested for determinism.

- **CPU vs GPU**
  - All core functionality runs on CPU.
  - Representation, scaling, and efficiency analyses will automatically use GPU if available, but can be forced onto CPU via config (`device: cpu`) or CLI arguments where applicable.

- **Deterministic runs**
  - The perturbation and dataset builders are deterministic for a fixed seed and configuration.
  - Deep learning training and inference are subject to standard PyTorch/transformers nondeterminism; for strict reproducibility, enable deterministic flags in your environment in addition to the seeds used here.

---

## Running Tests

The repository ships with a small, high-signal pytest suite that checks:

- pair generation invariants
- perturbation determinism
- tokenizer / representation / efficiency analysis shapes and sanity bounds

From the project root:

```powershell
python -m pytest
```

All tests:

- run on CPU
- avoid large model downloads by using small fake tokenizers/models
- complete in well under 60 seconds on a typical laptop

---

## Extended Experiments

The repository also includes a set of **extended experiments** and plotting
helpers intended for paper-ready analyses.

### Robustness scaling experiment

Run the larger FR–EN scaling experiment (using a broader model list and
larger dataset) with:

```powershell
python scripts/run_scaling_analysis.py --config configs/scaling_analysis_large.yaml
```

This writes:

- `results/tables/scaling_results_large.csv`

which contains, for each model:

- `model` / `model_name`
- `estimated_model_size` (in millions of parameters, when known)
- `clean_accuracy`
- `switchmix_accuracy`
- `robustness_drop`

### Cross-language robustness analysis

After you have generated robustness summaries for multiple language pairs
using `scripts/robustness_report.py` (e.g. FR–EN and FR–ES), you can build
a unified cross-language table:

```powershell
python scripts/run_cross_language_analysis.py
```

This reads all `results/tables/robustness_summary*.csv` files and writes:

- `results/tables/cross_language_robustness.csv`

with one row per `(language_pair, model)`.

### Plot generation

To generate publication-ready figures:

```powershell
python scripts/generate_plots.py
```

This creates:

- `results/figures/robustness_scaling.png`
  - x-axis: `estimated_model_size`
  - y-axis: `robustness_drop`
- `results/figures/cross_language_robustness.png`
  - x-axis: `language_pair`
  - y-axis: `robustness_drop`
  - colour: `model`

All figures are stored under:

- `results/figures/`

You can then include these directly in your paper or slides.

---

## Citation

If you use SwitchMixBench in academic work, please cite:

```bibtex
@misc{switchmixbench2026,
  title  = {SwitchMixBench: A Robustness Benchmark for Multilingual Foundation Models under Code-Switching},
  author = {Sidibe, Aissata},
  year   = {2026},
  note   = {Version 0.1},
  url    = {https://github.com/aissata-sidibe/switchmixbench}
}
```

---

## License

MIT License

