# SwitchMixBench

SwitchMixBench is a lightweight research benchmark for evaluating the robustness of multilingual foundation models under code-switching, informal language, and noisy text.

Rather than optimizing for leaderboard performance, the benchmark is designed to measure brittleness: how quickly performance collapses when inputs shift from clean text to realistic mixed-language user text.

---

## Motivation

Most multilingual evaluation benchmarks assume:
- clean, well-formed sentences
- a single language per example
- formal register

In practice, real-world user text often contains:
- code-switching
- informal spellings and abbreviations
- mixed scripts, borrowed words, and transliteration
- noisy punctuation and grammar

SwitchMixBench provides a reproducible way to quantify this gap.

---

## Tasks

Current tasks included:

- NLI (Natural Language Inference)  
  Given a premise and hypothesis, predict one label: entailment, neutral, or contradiction.

Additional tasks (QA / sentiment / safety) can be added using the same benchmark format.

---

## Data format

Each example includes:
- id: unique identifier
- task: task name (nli, qa, ...)
- split: train or test
- variant: clean or switchmix
- input / prompt: model input text
- label: ground truth label

The benchmark is structured so that clean vs switchmix examples share the same underlying meaning, but differ in surface form.

---

## Installation (Windows PowerShell)

Create a virtual environment:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

---

## Quickstart

Prepare processed benchmark JSON:

    python scripts/prepare_data.py

This generates:

    data/processed/switchmixbench.json

Run evaluation (NLI baseline):

    python switchmixbench/eval/run_eval.py --model google/flan-t5-small --task nli --split test

This writes:

    results/tables/nli_test_google_flan-t5-small.csv

---

## Research-grade (config-driven) pipeline

The original `scripts/prepare_data.py` + generative `run_eval.py` pipeline remains supported.
For research-scale experiments and reproducible analyses, use the YAML-driven dataset builder and analysis scripts below.

### Experimental protocol (recommended)

- **Build paired datasets**: generate `clean` vs `perturbed` examples at scale (5k–50k pairs per task/split).
- **Run robustness experiments**:
  - **Tokenizer-level**: how perturbations change tokenization statistics.
  - **Representation stability**: layer-wise cosine similarity clean vs perturbed hidden states.
  - **Scaling**: train (on clean) and evaluate clean vs perturbed across model sizes; report robustness gap \( \Delta \).
  - **Efficiency**: latency and sequence-length inflation under perturbations.
- **Save all outputs** under `results/tables/*.csv` for easy aggregation and plotting.

---

## Dataset building (JSONL, scalable)

Build a large paired dataset from a public Hugging Face dataset (with offline synthetic fallback):

    python scripts/build_dataset.py --config configs/dataset_xnli_fr_en.yaml

This generates (example):

    data/processed/nli_train_fr-en.jsonl
    data/processed/nli_test_fr-en.jsonl

Inspect dataset stats and example pairs:

    python scripts/inspect_dataset.py --path data/processed/nli_test_fr-en.jsonl --show 3

### Output schema (per JSONL row)

Each line is a JSON object with (at minimum):

- **task**: `nli` / `qa`
- **split**: `train` / `test`
- **variant**: `clean` / `perturbed`
- **input** (and `prompt` alias): model input text
- **label** (and `target` alias): ground-truth label/answer
- **metadata**: perturbation parameters and provenance

Additional fields (`pair_id`, `text_a`, `text_b`) are included to support analysis modules.

---

## Analysis modules (results/tables/*.csv)

All analysis scripts are independent CLI entry points and take YAML configs under `configs/`.

### Tokenization analysis

    python scripts/run_tokenizer_analysis.py --config configs/tokenizer_analysis.yaml

Writes:

    results/tables/tokenizer_stats.csv

### Representation stability (hidden-state drift)

    python scripts/run_representation_analysis.py --config configs/representation_analysis.yaml

Writes:

    results/tables/representation_shift.csv

### Multi-model scaling robustness (Δ gap across model sizes)

This fine-tunes each encoder model on **clean train** NLI data and evaluates on **clean vs perturbed test**.

    python scripts/run_scaling_analysis.py --config configs/scaling_analysis.yaml

Writes:

    results/tables/scaling_results.csv

### Efficiency impact

    python scripts/run_efficiency_analysis.py --config configs/efficiency_analysis.yaml

Writes:

    results/tables/efficiency_metrics.csv

---

## Reproducibility

- **All experiment knobs live in YAML** under `configs/` (dataset source, sizes, language pairs, perturbation intensity, analysis parameters).
- **Datasets are stored as JSONL** under `data/processed/` to preserve provenance and enable streaming.
- **Results are stored as CSV** under `results/tables/` for easy aggregation (`scripts/make_report.py` can still be used for concatenation).

### Expected outputs

- `data/processed/nli_{train,test}_fr-en.jsonl`
- `results/tables/tokenizer_stats.csv`
- `results/tables/representation_shift.csv`
- `results/tables/scaling_results.csv`
- `results/tables/efficiency_metrics.csv`

---

## Metrics

- NLI: Accuracy  
- QA: Token-level F1 (when enabled)

---

## Baseline results

We report performance under two conditions:
- clean: original inputs
- switchmix: code-switched + noisy variant

Model: google/flan-t5-small  
Task: NLI  
Metric: Accuracy  

Clean accuracy: 0.191  
SwitchMix accuracy: 0.030  
Robustness drop: 0.161  

Interpretation: the model exhibits a large degradation under code-switching noise, highlighting brittleness of multilingual foundation models under realistic mixed-language inputs.

---

## Robustness summary

Generate the robustness summary table:

    python scripts/robustness_report.py
    python -c "import pandas as pd; print(pd.read_csv('results/tables/robustness_summary.csv').to_string(index=False))"

Output:

    results/tables/robustness_summary.csv

---

## Reproducibility

- Raw datasets are not committed to the repo (data/raw/ is ignored).
- Results tables are generated by scripts and stored under results/tables/.
- Evaluation uses Hugging Face transformers for model loading and inference.

---

## Roadmap

Planned research extensions:
- add more multilingual foundation models (instruction-tuned + causal LMs)
- add more tasks (QA, sentiment, toxicity / safety)
- add controlled perturbations (typos, transliteration, slang)
- expand beyond French/English into West African languages (e.g., Bambara)

---

## Citation

If you use this benchmark, please cite:

    @misc{switchmixbench2026,
      title={SwitchMixBench: A Robustness Benchmark for Multilingual Foundation Models under Code-Switching},
      author={Aissata Sidibe},
      year={2026},
      url={https://github.com/aissata-sidibe/switchmixbench}
    }

---

## License

MIT License

