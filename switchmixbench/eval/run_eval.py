import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from switchmixbench.eval.metrics import accuracy, mean_f1
from switchmixbench.utils.io import read_json


def parse_nli_label(text: str) -> str:
    """
    Extract one of: entailment / neutral / contradiction from model output.
    This prevents 0% accuracy due to formatting like:
    'The answer is entailment.' or 'Entailment (because ...)'
    """
    t = str(text).lower().strip()
    t = t.replace(".", " ").replace(",", " ").replace(":", " ")
    t = " ".join(t.split())

    for lab in ["entailment", "neutral", "contradiction"]:
        if t == lab:
            return lab

    for lab in ["entailment", "neutral", "contradiction"]:
        if lab in t:
            return lab

    return ""


def load_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)

    # Try seq2seq first (e.g., FLAN-T5)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        kind = "seq2seq"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        kind = "causal"

    model.eval()
    return tok, model, kind


@torch.no_grad()
def generate(tok, model, kind: str, prompt: str, max_new_tokens: int = 32) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tok.decode(out[0], skip_special_tokens=True)

    # Some causal models echo the prompt
    if kind == "causal" and prompt in text:
        text = text.split(prompt, 1)[-1].strip()

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/switchmixbench.json")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["nli", "qa"], required=True)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit number of examples for quick debugging")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    os.makedirs("results/tables", exist_ok=True)

    data = read_json(args.data)

    # Filter by task + split
    rows = [r for r in data if r.get("task") == args.task and r.get("split") == args.split]

    # Optional debug subset
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    tok, model, kind = load_model(args.model)

    preds, labels, out_rows = [], [], []

    for r in tqdm(rows, desc=f"eval {args.task} {args.split}"):
        prompt = r.get("prompt", r.get("input"))
        if prompt is None:
            raise KeyError("Example missing both 'prompt' and 'input' keys.")

        pred_text = generate(tok, model, kind, prompt, max_new_tokens=args.max_new_tokens)

        # Store parsed prediction for scoring
        if args.task == "nli":
            preds.append(parse_nli_label(pred_text))
        else:
            preds.append(pred_text)

        labels.append(r["label"])

        # Save raw prediction text in output CSV
        out_rows.append(
            {
                **r,
                "prediction": pred_text,
                "model": args.model,
            }
        )

    df = pd.DataFrame(out_rows)

    if args.task == "nli":
        score = accuracy(preds, labels)
        metric = "accuracy"
    else:
        score = mean_f1(preds, labels)
        metric = "f1"

    print(f"{metric} = {score:.4f}")

    out_path = f"results/tables/{args.task}_{args.split}_{args.model.replace('/', '_')}.csv"
    df.to_csv(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    main()

