import os
import argparse
from switchmixbench.utils.io import write_json
from switchmixbench.generate.build_pairs import build_switchmix_pair
from switchmixbench.tasks.nli import format_nli_prompt
from switchmixbench.tasks.qa import format_qa_prompt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/processed/switchmixbench.json")
    args = p.parse_args()

    os.makedirs("data/processed", exist_ok=True)

    # v0.1 prototype dataset (small by design)
    seed_nli = [
        ("nli_0001", "Le chat est sur le tapis.", "The cat is on the rug.",
         "Un animal est sur le tapis.", "An animal is on the rug.", "entailment"),
        ("nli_0002", "Il pleut aujourd'hui à Paris.", "It is raining today in Paris.",
         "Le temps est sec.", "The weather is dry.", "contradiction"),
        ("nli_0003", "Marie a acheté une voiture.", "Marie bought a car.",
         "Marie aime le sport.", "Marie likes sports.", "neutral"),
    ]

    seed_qa = [
        ("qa_0001",
         "Paul est allé au marché ce matin pour acheter des fruits. Il a choisi des pommes et des bananes.",
         "Paul went to the market this morning to buy fruit. He chose apples and bananas.",
         "Quels fruits Paul a-t-il achetés ?",
         "What fruits did Paul buy?",
         "pommes et bananes"),
    ]

    examples = []

    # Build NLI
    for uid, fr_p, en_p, fr_h, en_h, label in seed_nli:
        exs = build_switchmix_pair(uid, "nli", "test", fr_p, en_p, fr_h, en_h, label=label)
        for e in exs:
            prompt = format_nli_prompt(e.text_a, e.text_b)
            examples.append({
                "uid": e.uid,
                "task": e.task,
                "split": e.split,
                "variant": e.variant,
                "prompt": prompt,
                "label": e.label,
            })

    # Build QA
    for uid, fr_ctx, en_ctx, fr_q, en_q, answer in seed_qa:
        exs = build_switchmix_pair(uid, "qa", "test", fr_ctx, en_ctx, fr_q, en_q, label=answer)
        for e in exs:
            prompt = format_qa_prompt(e.text_a, e.text_b)
            examples.append({
                "uid": e.uid,
                "task": e.task,
                "split": e.split,
                "variant": e.variant,
                "prompt": prompt,
                "label": e.label,
            })

    write_json(examples, args.out)
    print("saved:", args.out)
    print("n_examples:", len(examples))

if __name__ == "__main__":
    main()
