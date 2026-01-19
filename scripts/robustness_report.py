import os
import pandas as pd


def parse_nli_label(text: str) -> str:
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


def score_nli_accuracy(df: pd.DataFrame) -> float:
    preds = [parse_nli_label(x) for x in df["prediction"].astype(str).tolist()]
    labels = df["label"].astype(str).tolist()

    correct = 0
    for p, y in zip(preds, labels):
        if p == y:
            correct += 1

    return correct / max(1, len(labels))


def main():
    os.makedirs("results/tables", exist_ok=True)

    # We only summarize NLI for now (QA v0.2 will come next)
    path = "results/tables/nli_test_google_flan-t5-small.csv"
    if not os.path.exists(path):
        print("missing:", path)
        return

    df = pd.read_csv(path)

    df_clean = df[df["variant"] == "clean"]
    df_mix = df[df["variant"] == "switchmix"]

    clean_score = score_nli_accuracy(df_clean)
    mix_score = score_nli_accuracy(df_mix)

    out = pd.DataFrame(
        [
            {
                "task": "nli",
                "model": df["model"].iloc[0],
                "metric": "accuracy",
                "clean_score": clean_score,
                "switchmix_score": mix_score,
                "robustness_drop": clean_score - mix_score,
                "n_clean": len(df_clean),
                "n_switchmix": len(df_mix),
            }
        ]
    )

    out_path = "results/tables/robustness_summary.csv"
    out.to_csv(out_path, index=False)

    print("saved:", out_path)
    print(out)


if __name__ == "__main__":
    main()
