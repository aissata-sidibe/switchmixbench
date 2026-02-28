from __future__ import annotations

import sys
import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from switchmixbench.generate.perturbations import apply_switchmix_perturbation
from switchmixbench.generate.synthetic import generate_synthetic_nli, generate_synthetic_qa
from switchmixbench.tasks.nli import format_nli_prompt
from switchmixbench.tasks.qa import format_qa_prompt
from switchmixbench.utils.config import get, load_yaml
from switchmixbench.utils.io import write_jsonl


def _lazy_load_dataset():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face dataset loading requires `datasets`. Install with: pip install datasets"
        ) from e
    return load_dataset


LABEL_MAP_XNLI = {0: "entailment", 1: "neutral", 2: "contradiction"}


def _sample_indices(n_total: int, n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    idx = list(range(n_total))
    rng.shuffle(idx)
    return idx[:n]


def _build_rows_from_nli_pair(
    pair_id: str,
    split: str,
    premise_clean: str,
    premise_insert: str,
    hyp_clean: str,
    hyp_insert: str,
    label: str,
    clean_lang: str,
    insert_lang: str,
    perturb: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    p_switch = float(get(perturb, "code_switch.p_switch", 0.35))
    noise_p = float(get(perturb, "noise.p", 0.15))

    prem_pert, prem_meta = apply_switchmix_perturbation(
        premise_clean, premise_insert, seed=seed, p_switch=p_switch, noise_p=noise_p
    )
    hyp_pert, hyp_meta = apply_switchmix_perturbation(
        hyp_clean, hyp_insert, seed=seed + 7, p_switch=p_switch, noise_p=noise_p
    )

    clean_prompt = format_nli_prompt(premise_clean, hyp_clean)
    pert_prompt = format_nli_prompt(prem_pert, hyp_pert)

    base_meta = {
        "pair_id": pair_id,
        "task": "nli",
        "languages": {"clean": clean_lang, "insert": insert_lang},
    }

    clean_row = {
        "id": f"{pair_id}__clean",
        "pair_id": pair_id,
        "task": "nli",
        "split": split,
        "variant": "clean",
        "input": clean_prompt,
        "prompt": clean_prompt,
        "text_a": premise_clean,
        "text_b": hyp_clean,
        "label": label,
        "target": label,
        "metadata": {**base_meta, "perturbation": {"type": "none"}},
    }

    pert_row = {
        "id": f"{pair_id}__perturbed",
        "pair_id": pair_id,
        "task": "nli",
        "split": split,
        "variant": "perturbed",
        "input": pert_prompt,
        "prompt": pert_prompt,
        "text_a": prem_pert,
        "text_b": hyp_pert,
        "label": label,
        "target": label,
        "metadata": {
            **base_meta,
            "perturbation": {
                "type": "switchmix+informal_noise",
                "premise": prem_meta,
                "hypothesis": hyp_meta,
                "params": {"p_switch": p_switch, "noise_p": noise_p},
            },
        },
    }

    return [clean_row, pert_row]


def _build_rows_from_qa_pair(
    pair_id: str,
    split: str,
    ctx_clean: str,
    ctx_insert: str,
    q_clean: str,
    q_insert: str,
    answer: str,
    clean_lang: str,
    insert_lang: str,
    perturb: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    p_switch = float(get(perturb, "code_switch.p_switch", 0.35))
    noise_p = float(get(perturb, "noise.p", 0.15))

    ctx_pert, ctx_meta = apply_switchmix_perturbation(
        ctx_clean, ctx_insert, seed=seed, p_switch=p_switch, noise_p=noise_p
    )
    q_pert, q_meta = apply_switchmix_perturbation(
        q_clean, q_insert, seed=seed + 7, p_switch=p_switch, noise_p=noise_p
    )

    clean_prompt = format_qa_prompt(ctx_clean, q_clean)
    pert_prompt = format_qa_prompt(ctx_pert, q_pert)

    base_meta = {
        "pair_id": pair_id,
        "task": "qa",
        "languages": {"clean": clean_lang, "insert": insert_lang},
    }

    clean_row = {
        "id": f"{pair_id}__clean",
        "pair_id": pair_id,
        "task": "qa",
        "split": split,
        "variant": "clean",
        "input": clean_prompt,
        "prompt": clean_prompt,
        "text_a": ctx_clean,
        "text_b": q_clean,
        "label": answer,
        "target": answer,
        "metadata": {**base_meta, "perturbation": {"type": "none"}},
    }

    pert_row = {
        "id": f"{pair_id}__perturbed",
        "pair_id": pair_id,
        "task": "qa",
        "split": split,
        "variant": "perturbed",
        "input": pert_prompt,
        "prompt": pert_prompt,
        "text_a": ctx_pert,
        "text_b": q_pert,
        "label": answer,
        "target": answer,
        "metadata": {
            **base_meta,
            "perturbation": {
                "type": "switchmix+informal_noise",
                "context": ctx_meta,
                "question": q_meta,
                "params": {"p_switch": p_switch, "noise_p": noise_p},
            },
        },
    }

    return [clean_row, pert_row]


def build_from_hf(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    task = str(get(cfg, "task"))
    seed = int(get(cfg, "seed", 42))
    perturb = get(cfg, "perturbations", {}) or {}

    clean_lang = str(get(cfg, "language_pair.clean_lang", "fr"))
    insert_lang = str(get(cfg, "language_pair.insert_lang", "en"))

    dataset_name = str(get(cfg, "source.dataset_name"))
    source_splits = get(cfg, "splits", {}) or {}

    load_dataset = _lazy_load_dataset()

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}

    if task == "nli":
        ds_clean = load_dataset(dataset_name, clean_lang)
        ds_insert = load_dataset(dataset_name, insert_lang)

        for out_split, spec in source_splits.items():
            src_split = str(spec.get("source_split", out_split))
            n = int(spec.get("n_examples", 5000))

            split_clean = ds_clean[src_split]
            split_insert = ds_insert[src_split]
            n_total = min(len(split_clean), len(split_insert))
            idxs = _sample_indices(n_total, min(n, n_total), seed=seed + hash(out_split) % 10_000)

            out_rows: List[Dict[str, Any]] = []
            for i in idxs:
                ex_c = split_clean[i]
                ex_i = split_insert[i]
                pair_id = str(ex_c.get("id") or f"{dataset_name}_{clean_lang}_{src_split}_{i}")
                label = LABEL_MAP_XNLI.get(int(ex_c["label"]), "neutral")

                out_rows.extend(
                    _build_rows_from_nli_pair(
                        pair_id=pair_id,
                        split=out_split,
                        premise_clean=str(ex_c["premise"]),
                        premise_insert=str(ex_i["premise"]),
                        hyp_clean=str(ex_c["hypothesis"]),
                        hyp_insert=str(ex_i["hypothesis"]),
                        label=label,
                        clean_lang=clean_lang,
                        insert_lang=insert_lang,
                        perturb=perturb,
                        seed=seed + i,
                    )
                )

            rows_by_split[out_split] = out_rows

    elif task == "qa":
        # Default HF QA source: xquad (language configs like "xquad.en", "xquad.fr")
        # Config should set dataset_name and dataset_config_clean/insert if needed.
        ds_cfg_clean = get(cfg, "source.dataset_config_clean", clean_lang)
        ds_cfg_insert = get(cfg, "source.dataset_config_insert", insert_lang)
        ds_clean = load_dataset(dataset_name, ds_cfg_clean)
        ds_insert = load_dataset(dataset_name, ds_cfg_insert)

        for out_split, spec in source_splits.items():
            src_split = str(spec.get("source_split", out_split))
            n = int(spec.get("n_examples", 5000))

            split_clean = ds_clean[src_split]
            split_insert = ds_insert[src_split]
            n_total = min(len(split_clean), len(split_insert))
            idxs = _sample_indices(n_total, min(n, n_total), seed=seed + hash(out_split) % 10_000)

            out_rows: List[Dict[str, Any]] = []
            for i in idxs:
                ex_c = split_clean[i]
                ex_i = split_insert[i]
                pair_id = str(ex_c.get("id") or f"{dataset_name}_{ds_cfg_clean}_{src_split}_{i}")

                # Most HF QA datasets store answers as dict with "text" list.
                ans = ex_c.get("answers", {})
                if isinstance(ans, dict) and "text" in ans and ans["text"]:
                    answer = str(ans["text"][0])
                else:
                    # Fallback: try direct field
                    answer = str(ex_c.get("answer", ""))

                out_rows.extend(
                    _build_rows_from_qa_pair(
                        pair_id=pair_id,
                        split=out_split,
                        ctx_clean=str(ex_c["context"]),
                        ctx_insert=str(ex_i["context"]),
                        q_clean=str(ex_c["question"]),
                        q_insert=str(ex_i["question"]),
                        answer=answer,
                        clean_lang=clean_lang,
                        insert_lang=insert_lang,
                        perturb=perturb,
                        seed=seed + i,
                    )
                )

            rows_by_split[out_split] = out_rows
    else:
        raise ValueError(f"Unknown task: {task}")

    return rows_by_split


def build_synthetic(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    task = str(get(cfg, "task"))
    seed = int(get(cfg, "seed", 42))
    perturb = get(cfg, "perturbations", {}) or {}

    clean_lang = str(get(cfg, "language_pair.clean_lang", "fr"))
    insert_lang = str(get(cfg, "language_pair.insert_lang", "en"))

    splits = get(cfg, "splits", {}) or {"test": {"n_examples": 5000}}
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for out_split, spec in splits.items():
        n = int(spec.get("n_examples", 5000))
        split_seed = seed + hash(out_split) % 10_000
        out_rows: List[Dict[str, Any]] = []

        if task == "nli":
            pairs = generate_synthetic_nli(n=n, seed=split_seed)
            for j, p in enumerate(pairs):
                out_rows.extend(
                    _build_rows_from_nli_pair(
                        pair_id=p.pair_id,
                        split=out_split,
                        premise_clean=p.premise_fr,
                        premise_insert=p.premise_en,
                        hyp_clean=p.hypothesis_fr,
                        hyp_insert=p.hypothesis_en,
                        label=p.label,
                        clean_lang=clean_lang,
                        insert_lang=insert_lang,
                        perturb=perturb,
                        seed=split_seed + j,
                    )
                )
        elif task == "qa":
            pairs = generate_synthetic_qa(n=n, seed=split_seed)
            for j, p in enumerate(pairs):
                out_rows.extend(
                    _build_rows_from_qa_pair(
                        pair_id=p.pair_id,
                        split=out_split,
                        ctx_clean=p.context_fr,
                        ctx_insert=p.context_en,
                        q_clean=p.question_fr,
                        q_insert=p.question_en,
                        answer=p.answer,
                        clean_lang=clean_lang,
                        insert_lang=insert_lang,
                        perturb=perturb,
                        seed=split_seed + j,
                    )
                )
        else:
            raise ValueError(f"Unknown task: {task}")

        rows_by_split[out_split] = out_rows

    return rows_by_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config under configs/")
    ap.add_argument("--out_dir", type=str, default=None, help="Override output directory")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    out_dir = Path(args.out_dir or get(cfg, "output.out_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    source_kind = str(get(cfg, "source.kind", "hf")).lower()
    allow_fallback = bool(get(cfg, "source.allow_fallback_synthetic", True))

    try:
        if source_kind == "hf":
            rows_by_split = build_from_hf(cfg)
        elif source_kind == "synthetic":
            rows_by_split = build_synthetic(cfg)
        else:
            raise ValueError(f"Unknown source.kind: {source_kind}")
    except Exception as e:
        if not allow_fallback:
            raise
        print("[build_dataset] HF source failed; falling back to synthetic generation.")
        print("[build_dataset] Error:", repr(e))
        rows_by_split = build_synthetic(cfg)

    task = str(get(cfg, "task"))
    clean_lang = str(get(cfg, "language_pair.clean_lang", "fr"))
    insert_lang = str(get(cfg, "language_pair.insert_lang", "en"))

    for split, rows in rows_by_split.items():
        out_path = out_dir / f"{task}_{split}_{clean_lang}-{insert_lang}.jsonl"
        write_jsonl(rows, out_path)
        print("saved:", out_path, "n_rows:", len(rows))


if __name__ == "__main__":
    main()

