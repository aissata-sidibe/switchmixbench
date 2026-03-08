## Adding a new language pair to SwitchMixBench

This document describes how to add a new language pair (e.g. FR–ES) to the
SwitchMixBench pipeline in a **minimal, reproducible** way.

The key idea is to reuse the existing **paired clean/perturbed protocol**
and plug in a new base/insert language pair.

---

### 1. Choose a source dataset

For NLI, we recommend using **XNLI** via Hugging Face Datasets:

- Base language: e.g. `fr`
- Insert language: e.g. `es`

You can mirror the existing FR–EN configs:

- `configs/dataset_xnli_fr_en.yaml`

and create a new one:

- `configs/dataset_xnli_fr_es.yaml`

with:

```yaml
seed: 42

task: nli

language_pair:
  clean_lang: fr
  insert_lang: es

source:
  kind: hf
  dataset_name: xnli
  allow_fallback_synthetic: true

splits:
  train:
    source_split: train
    n_examples: 20000
  test:
    source_split: validation
    n_examples: 5000

perturbations:
  code_switch:
    p_switch: 0.35
  noise:
    p: 0.15

output:
  out_dir: data/processed
```

The dataset builder will then load:

- `xnli` with config `"fr"` for the base language, and
- `xnli` with config `"es"` for the insert language.

---

### 2. Reuse the perturbation API

The core perturbation logic is language-agnostic:

- `switchmixbench.generate.code_switch_rules.mix_sentences`
  - Treats the first argument as the base sentence (e.g. FR) and the
    second as the insert sentence (e.g. ES).
- `switchmixbench.generate.noise_injectors.inject_informal_noise`
  - Currently tuned for FR/EN but still safe to use as a minimal informal
    drift for other languages.
- `switchmixbench.generate.perturbations.apply_switchmix_perturbation`
  - Composes code-switching and informal noise and returns structured
    metadata.

For a new pair you can:

1. Keep using the same functions.
2. Optionally extend them with **language‑specific rules**:
   - e.g. add a small list of common ES discourse markers for insertion,
   - introduce ES‑specific abbreviations.

When adding such rules, keep them:

- small and well‑documented
- deterministic for a fixed seed (tests rely on this)

---

### 3. Build the new language-pair dataset

Once the config is in place, build the processed JSONL files:

```bash
python scripts/build_dataset.py --config configs/dataset_xnli_fr_es.yaml
```

You should see outputs such as:

- `data/processed/nli_train_fr-es.jsonl`
- `data/processed/nli_test_fr-es.jsonl`

Each row will contain:

- `task`, `split`, `variant` (`clean` / `perturbed`)
- `input` / `prompt`
- `label` / `target`
- `pair_id`
- `metadata.languages.clean` / `metadata.languages.insert`

---

### 4. Run analyses on the new pair

To analyse the new language pair, point the analysis configs to the new
`data_paths`. For example, you can clone:

- `configs/tokenizer_analysis.yaml` into `configs/tokenizer_analysis_fr_es.yaml`
- `configs/representation_analysis.yaml` into `configs/representation_analysis_fr_es.yaml`
- `configs/efficiency_analysis.yaml` into `configs/efficiency_analysis_fr_es.yaml`

and update:

```yaml
data_paths:
  - data/processed/nli_test_fr-es.jsonl
```

Then run:

```bash
python scripts/run_tokenizer_analysis.py --config configs/tokenizer_analysis_fr_es.yaml
python scripts/run_representation_analysis.py --config configs/representation_analysis_fr_es.yaml
python scripts/run_efficiency_analysis.py --config configs/efficiency_analysis_fr_es.yaml
```

All results will still be written under `results/tables/`, but the
`data_path` column in each CSV will include the language pair, making it
easy to filter for FR–ES vs FR–EN.

---

### 5. Recommended best practices

- **Keep configs small and explicit**
  - Prefer adding a new YAML file over overloading a single one with many
    conditionals.
- **Document new rules**
  - When adding language‑specific lexicons or noise rules, include a short
    comment explaining their intent and limitations.
- **Test determinism**
  - If you extend the perturbation logic, add a small pytest that asserts
    determinism for a fixed seed (mirroring the existing tests).

Following this pattern ensures new language pairs slot cleanly into the
benchmark without affecting existing experiments or CSV schemas.

