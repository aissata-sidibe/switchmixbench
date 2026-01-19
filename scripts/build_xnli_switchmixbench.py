import os
import json
import random

RAW_PATH = "data/raw/xnli_fr_1000.json"
OUT_PATH = "data/processed/switchmixbench.json"

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

EN_INSERTIONS = [
    "to be honest",
    "in my opinion",
    "I mean",
    "basically",
    "you know",
    "for real",
    "actually",
]

def make_clean_prompt(premise: str, hypothesis: str) -> str:
    return (
        "Tâche: inférence en langage naturel (NLI).\n"
        "Détermine si l'hypothèse est une conséquence logique de la prémisse.\n\n"
        f"Prémisse: {premise}\n"
        f"Hypothèse: {hypothesis}\n\n"
        "Réponds avec une seule étiquette: entailment, neutral, contradiction."
    )

def make_switchmix_prompt(premise: str, hypothesis: str) -> str:
    phrase = random.choice(EN_INSERTIONS)
    return (
        "Task: Natural Language Inference.\n"
        "Tu dois décider si l'hypothèse suit logiquement la prémisse.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"({phrase}) Réponds avec: entailment, neutral, contradiction."
    )

def main():
    random.seed(42)

    os.makedirs("data/processed", exist_ok=True)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = []

    for ex in raw:
        label = LABEL_MAP.get(ex["label"], "neutral")

        out.append({
            "id": ex["id"] + "_clean",
            "task": "nli",
            "split": "test",
            "variant": "clean",
            "input": make_clean_prompt(ex["premise"], ex["hypothesis"]),
            "label": label,
        })

        out.append({
            "id": ex["id"] + "_switchmix",
            "task": "nli",
            "split": "test",
            "variant": "switchmix",
            "input": make_switchmix_prompt(ex["premise"], ex["hypothesis"]),
            "label": label,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"saved: {OUT_PATH}")
    print(f"n_examples: {len(out)}")
    print("example:", out[0]["id"])

if __name__ == "__main__":
    main()
