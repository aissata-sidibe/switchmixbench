# SwitchMixBench Dataset Card (v0.1)

## Summary
SwitchMixBench is a small evaluation benchmark designed to measure robustness of multilingual foundation models to
code-switching and informal language (FR↔EN in v0.1).

## Tasks
- NLI (entailment/contradiction/neutral)
- QA (short answer)

## Variants
- clean: monolingual French prompts
- switchmix: code-switched + informal noise

## Intended Use
Research benchmarking, robustness evaluation, and failure-mode analysis.

## Limitations
v0.1 is a small prototype dataset. Future versions will scale using curated sources and stronger validation.
