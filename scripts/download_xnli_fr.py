from datasets import load_dataset
import random
import os
import json

OUT_PATH = "data/raw/xnli_fr_1000.json"

def main():
    os.makedirs("data/raw", exist_ok=True)

    ds = load_dataset("xnli", "fr")

    # Use validation split (cleaner + stable)
    split = ds["validation"]

    # Shuffle deterministically
    random.seed(42)
    indices = list(range(len(split)))
    random.shuffle(indices)

    n = 1000
    indices = indices[:n]

    examples = []
    for i in indices:
        ex = split[i]
        examples.append({
            "id": f"xnli_fr_val_{i}",
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "label": ex["label"],  # 0 entailment, 1 neutral, 2 contradiction
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"saved: {OUT_PATH}")
    print(f"n_examples: {len(examples)}")

if __name__ == "__main__":
    main()
