import glob
import os
import pandas as pd

def main():
    os.makedirs("results/tables", exist_ok=True)
    files = glob.glob("results/tables/*.csv")
    if not files:
        raise RuntimeError("No results found. Run eval first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv("results/tables/all_results.csv", index=False)
    print("saved: results/tables/all_results.csv")

if __name__ == "__main__":
    main()
