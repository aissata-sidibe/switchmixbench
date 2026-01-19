# SwitchMixBench (v0.1)
**Code-switching robustness benchmark for multilingual foundation models (FR/EN)**

SwitchMixBench is a lightweight benchmark designed to measure how multilingual foundation models degrade under:
- **code-switching** (French ↔ English)
- **informal language**
- mild **distribution shift** in prompt phrasing

The goal is to support reproducible robustness evaluation and failure-mode analysis for multilingual NLP.

---

## Research question
**How robust are multilingual foundation models to realistic code-switching and informal language, compared to clean monolingual inputs?**

---

## Tasks (v0.1)
- **NLI**: entailment / contradiction / neutral  
- **QA**: short-answer question answering

Each example has two variants:
- `clean` (monolingual French)
- `switchmix` (French + English span insertion + informal noise)

---

## Quickstart (Windows / CPU)
### 1) Install
```bash
pip install -r requirements.txt
pip install -e .

@misc{switchmixbench2026,
  title={SwitchMixBench: Code-switching Robustness Benchmark for Multilingual Foundation Models},
  author={Aissata Sidibe},
  year={2026},
  url={https://github.com/aissata-sidibe/switchmixbench}
}

