## SwitchMixBench Benchmark Design

### Paired evaluation protocol

SwitchMixBench is built around **paired examples**:

- Every semantic example is represented by two surface forms:
  - a **clean** variant in a single language
  - a **perturbed** variant that introduces code-switching and light noise
- Both variants share:
  - the same **underlying meaning**
  - the same **label / answer**
  - a shared `pair_id` used by analysis modules

This enables **brittleness measurements** that compare model behaviour under
distribution shift while holding the ground truth fixed.

For each pair, we can compute:

- performance drop (e.g. accuracy, F1)
- tokenisation changes (sequence length, fragmentation)
- hidden-state drift (cosine distance across layers)
- efficiency impact (latency and activation size)

### Perturbation types

The current benchmark focuses on **minimal, transparent perturbations**:

- **Code-switching**
  - A short span from a second language is inserted into the base sentence.
  - Implemented in `switchmixbench.generate.code_switch_rules.mix_sentences`.
- **Informal noise**
  - Drops French negation tokens occasionally (`ne`, `pas`).
  - Abbreviates some English tokens (`you`→`u`, `are`→`r`).
  - Adds light punctuation / emoticons.
  - Implemented in `switchmixbench.generate.noise_injectors.inject_informal_noise`.

The composed perturbation with metadata is defined in:

- `switchmixbench.generate.perturbations.apply_switchmix_perturbation`

This design keeps perturbations **simple enough to reason about**, while
still capturing realistic failure modes for multilingual LMs.

### Analyses

SwitchMixBench includes several analysis modules under `switchmixbench/analysis`:

- **Tokeniser analysis** (`tokenizer_analysis.py`)
  - Measures token count inflation, fragmentation, tokens-per-character and
    token distribution shift (JS divergence) between clean vs perturbed.
- **Representation stability** (`representation_analysis.py`)
  - Extracts hidden states across all layers of an encoder model.
  - Computes cosine similarity and drift between clean / perturbed pairs.
- **Scaling analysis** (`scaling_analysis.py`)
  - Fine-tunes multiple encoder models on clean NLI data.
  - Evaluates each model on both clean and perturbed test splits.
  - Reports the robustness gap Δ = accuracy(clean) − accuracy(perturbed).
- **Efficiency analysis** (`efficiency_analysis.py`)
  - Measures average sequence length, latency and a simple activation memory
    proxy for clean vs perturbed inputs.

All analyses write tidy CSV tables to `results/tables/` for easy inspection.

### Recommended extensions

The repository is intentionally small and modular to make extensions easy.
A few natural research directions:

- **More tasks**
  - Extend from NLI to QA, sentiment, toxicity, safety, etc.
  - Add task-specific metrics while keeping the paired structure.
- **More languages**
  - Add additional language pairs (e.g. FR–ES, EN–ES, FR–AR).
  - Use the same perturbation API with language-specific rules.
- **Richer perturbations**
  - Introduce typos, transliteration, slang dictionaries, emoji.
  - Compose multiple perturbation types in a controlled way.
- **Mechanistic analyses**
  - Probe layer-wise attention patterns under perturbations.
  - Study subnetwork or neuron-level responses to code-switching.

The `docs/adding_language_pairs.md` guide provides a concrete recipe for
adding new language pairs while staying consistent with the current design.

