import argparse
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

from switchmixbench.eval.metrics import accuracy, mean_f1

from switchmixbench.utils.io import read_json

def load_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        kind = "seq2seq"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        kind = "causal"

    model.eval()
    return tok, model, kind

@torch.no_grad()
def generate(tok, model, kind, prompt: str, max_new_tokens=32):
    inputs = tok(prompt, return_tensors="pt", truncation=True)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tok.decode(out[0], skip_special_tokens=True)

    if kind == "causal" and prompt in text:
        text = text.split(prompt, 1)[-1].strip()

    return text.strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/switchmixbench.json")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--task", type=str, choices=["nli", "qa"], required=True)
    p.add_argument("--split", type=str, choices=["train", "test"], default="test")
    args = p.parse_args()

    data = read_json(args.data)
    rows = [r for r in data if r["task"] == args.task and r["split"] == args.split]

    tok, model, kind = load_model(args.model)

    preds, labels, out_rows = [], [], []
    for r in tqdm(rows, desc=f"eval {args.task} {args.split}"):
        prompt = r["prompt"]
        pred = generate(tok, model, kind, prompt)
        preds.append(pred)
        labels.append(r["label"])
        out_rows.append({**r, "prediction": pred, "model": args.model})

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
