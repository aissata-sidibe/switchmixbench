import os
import pandas as pd

def normalize(x):
    return str(x).strip().lower()

def score_nli(df):
    # accuracy
    correct = 0
    for _, r in df.iterrows():
        if normalize(r["prediction"]) == normalize(r["label"]):
            correct += 1
    return correct / max(1, len(df))

def score_qa(df):
    # exact match
    correct = 0
    for _, r in df.iterrows():
        if normalize(r["prediction"]) == normalize(r["label"]):
            correct += 1
    return correct / max(1, len(df))

def main():
    os.makedirs("results/tables", exist_ok=True)

    files = [
        "results/tables/nli_test_google_flan-t5-small.csv",
        "results/tables/qa_test_google_flan-t5-small.csv",
    ]

    rows = []

    for path in files:
        if not os.path.exists(path):
            print("missing:", path)
            continue

        df = pd.read_csv(path)

        task = df["task"].iloc[0]
        model = df["model"].iloc[0]

        df_clean = df[df["variant"] == "clean"]
        df_mix = df[df["variant"] == "switchmix"]

        if task == "nli":
            clean_score = score_nli(df_clean)
            mix_score = score_nli(df_mix)
            metric = "accuracy"
        else:
            clean_score = score_qa(df_clean)
            mix_score = score_qa(df_mix)
            metric = "exact_match"

        rows.append({
            "task": task,
            "model": model,
            "metric": metric,
            "clean_score": clean_score,
            "switchmix_score": mix_score,
            "robustness_drop": clean_score - mix_score
        })

    out = pd.DataFrame(rows)
    out_path = "results/tables/robustness_summary.csv"
    out.to_csv(out_path, index=False)
    print("saved:", out_path)
    print(out)

if __name__ == "__main__":
    main()
