from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from switchmixbench.utils.io import read_any


def _lazy_torch_transformers():
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Scaling analysis requires torch + transformers."
        ) from e
    return (
        torch,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )


LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _extract_pair_texts(row: Dict[str, Any]) -> Tuple[str, str]:
    a = row.get("text_a")
    b = row.get("text_b")
    if a is not None and b is not None:
        return str(a), str(b)

    # Prompt parsing fallback (best-effort, supports v0.1 NLI prompt)
    prompt = str(row.get("input") or row.get("prompt") or "")
    prem = ""
    hyp = ""
    for line in prompt.splitlines():
        if line.strip().lower().startswith("premise:"):
            prem = line.split(":", 1)[-1].strip()
        if line.strip().lower().startswith("hypothesis:"):
            hyp = line.split(":", 1)[-1].strip()
    return prem, hyp


def _load_rows(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        obj = read_any(p)
        if isinstance(obj, list):
            rows.extend(obj)
        else:
            raise ValueError(f"Expected list rows in {p}")
    return rows


def _split_variant(rows: List[Dict[str, Any]], split: str, variant: str, task: str = "nli") -> List[Dict[str, Any]]:
    return [
        r
        for r in rows
        if str(r.get("task")) == task and str(r.get("split")) == split and str(r.get("variant")) == variant
    ]


class _NLITorchDataset:
    def __init__(self, encodings: Dict[str, Any], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _prepare_nli_dataset(tok, rows: List[Dict[str, Any]], max_length: int) -> Tuple[_NLITorchDataset, int]:
    text_a = []
    text_b = []
    labels = []
    skipped = 0
    for r in rows:
        prem, hyp = _extract_pair_texts(r)
        lab = str(r.get("label", "")).strip().lower()
        if not prem or not hyp or lab not in LABEL2ID:
            skipped += 1
            continue
        text_a.append(prem)
        text_b.append(hyp)
        labels.append(LABEL2ID[lab])

    enc = tok(
        text_a,
        text_b,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    ds = _NLITorchDataset(enc, labels)
    return ds, skipped


def _accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argmax(axis=-1)
    return float((preds == labels).mean()) if len(labels) else 0.0


def run_scaling_experiment(
    train_paths: List[str],
    test_paths: List[str],
    model_names: List[str],
    out_csv: str = "results/tables/scaling_results.csv",
    seed: int = 42,
    max_length: int = 256,
    train_split: str = "train",
    test_split: str = "test",
    train_variant: str = "clean",
    test_clean_variant: str = "clean",
    test_pert_variant: str = "perturbed",
    max_train_examples: Optional[int] = None,
    max_test_examples: Optional[int] = None,
    epochs: float = 1.0,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> str:
    (
        torch,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    ) = _lazy_torch_transformers()

    set_seed(seed)

    train_rows = _load_rows(train_paths)
    test_rows = _load_rows(test_paths)

    train_rows = _split_variant(train_rows, split=train_split, variant=train_variant, task="nli")
    test_clean = _split_variant(test_rows, split=test_split, variant=test_clean_variant, task="nli")
    test_pert = _split_variant(test_rows, split=test_split, variant=test_pert_variant, task="nli")

    if max_train_examples is not None:
        train_rows = train_rows[: max_train_examples]
    if max_test_examples is not None:
        test_clean = test_clean[: max_test_examples]
        test_pert = test_pert[: max_test_examples]

    records: List[Dict[str, Any]] = []

    for model_name in model_names:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        train_ds, train_skipped = _prepare_nli_dataset(tok, train_rows, max_length=max_length)
        test_clean_ds, test_clean_skipped = _prepare_nli_dataset(tok, test_clean, max_length=max_length)
        test_pert_ds, test_pert_skipped = _prepare_nli_dataset(tok, test_pert, max_length=max_length)

        data_collator = DataCollatorWithPadding(tokenizer=tok)

        args = TrainingArguments(
            output_dir="results/artifacts/scaling_tmp",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=False,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            seed=seed,
        )

        trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=data_collator, tokenizer=tok)

        t0 = time.time()
        trainer.train()
        train_time_s = time.time() - t0

        # Predict clean vs perturbed
        pred_clean = trainer.predict(test_clean_ds)
        pred_pert = trainer.predict(test_pert_ds)

        clean_acc = _accuracy_from_logits(pred_clean.predictions, pred_clean.label_ids)
        pert_acc = _accuracy_from_logits(pred_pert.predictions, pred_pert.label_ids)

        records.append(
            {
                "task": "nli",
                "model": model_name,
                "seed": int(seed),
                "train_split": train_split,
                "test_split": test_split,
                "n_train": int(len(train_ds)),
                "n_test_clean": int(len(test_clean_ds)),
                "n_test_perturbed": int(len(test_pert_ds)),
                "train_skipped": int(train_skipped),
                "test_clean_skipped": int(test_clean_skipped),
                "test_perturbed_skipped": int(test_pert_skipped),
                "clean_accuracy": float(clean_acc),
                "perturbed_accuracy": float(pert_acc),
                "robustness_gap_delta": float(clean_acc - pert_acc),
                "epochs": float(epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "max_length": int(max_length),
                "train_time_s": float(train_time_s),
            }
        )

    df = pd.DataFrame.from_records(records)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)

