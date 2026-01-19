# SwitchMixBench
**A robustness benchmark for multilingual foundation models under code-switching and informal language (FR↔EN)**

SwitchMixBench is an open-source benchmark designed to evaluate how multilingual **Large Language Models (LLMs)** and other **foundation models** behave under realistic distribution shifts such as **code-switching** and informal writing. The benchmark pairs clean inputs with perturbed variants and reports robustness gaps across tasks.

This repository provides:
- a small benchmark dataset (v0.1 prototype)
- a reproducible evaluation harness
- baseline + foundation model evaluation scripts
- robustness reporting utilities

---

## Motivation
Multilingual evaluations often assume clean monolingual text. In practice, users frequently mix languages (e.g., French↔English), write informally, and include borrowed expressions. These phenomena can cause silent failures in reasoning, entailment judgments, and answer extraction.

SwitchMixBench focuses on *controlled perturbations* to quantify robustness and support failure-mode analysis.

---

## Research Question
**How robust are multilingual foundation models to realistic code-switching and informal language compared to clean monolingual inputs?**

### Hypothesis (v0.1)
Performance on clean text will not reliably predict performance under code-switching + informality, and failure modes will vary by task (NLI vs QA).

---

## Tasks (v0.1)
SwitchMixBench currently includes:

### 1) Natural Language Inference (NLI)
- Labels: `entailment`, `contradiction`, `neutral`
- Metric: **Accuracy**

### 2) Question Answering (QA)
- Short-answer QA from a context + question prompt
- Metric: **Token-level F1** (SQuAD-style)

---

## Benchmark Variants
Each example is paired across two input conditions:

- `clean`: monolingual French prompt
- `switchmix`: code-switched prompt (FR↔EN span insertion) + informal noise injection

This enables direct measurement of robustness drop under controlled shift.

---

## Repository Structure
```text
switchmixbench/
├── data/
│   └── dataset_card.md
├── scripts/
│   ├── prepare_data.py
│   ├── run_baselines.py
│   ├── robustness_report.py
│   └── make_report.py
├── switchmixbench/
│   ├── generate/
│   ├── tasks/
│   ├── eval/
│   └── utils/
├── requirements.txt
├── pyproject.toml
└── README.md
---
## Results (baseline)

We report accuracy on NLI under two conditions:

- **clean**: original English NLI inputs
- **switchmix**: code-switched + noisy variant (SwitchMix)

| Model | Task | Metric | Clean | SwitchMix | Robustness Drop |
|------|------|--------|------:|----------:|----------------:|
| google/flan-t5-small | NLI | Accuracy | 0.191 | 0.030 | 0.161 |

**Observation:** The model suffers a large degradation under code-switching, highlighting brittleness of multilingual foundation models under realistic mixed-language inputs.

