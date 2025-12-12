import argparse
import os
from typing import Dict

import pandas as pd


def load_training_data(rows: int) -> Dict[str, pd.DataFrame]:
    # Scan top-level TrainingData + pools
    training_base = os.path.join(os.getcwd(), "TrainingData")
    pools_base = os.path.join(training_base, "Fixed")
    out: Dict[str, pd.DataFrame] = {}

    # Top-level CSVs commonly present in TrainingData
    for fname in ("training_data.csv", "test_data.csv"):
        path = os.path.join(training_base, fname)
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
                if rows and len(df) > rows:
                    df = df.head(rows)
                out[fname] = df
            except Exception as e:
                out[fname] = pd.DataFrame({"error": [str(e)], "file": [path]})

    # JSONL (e.g., friday_preferences.jsonl)
    jsonl_path = os.path.join(training_base, "friday_preferences.jsonl")
    if os.path.isfile(jsonl_path):
        try:
            records = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        import json
                        obj = json.loads(line)
                    except Exception as je:
                        obj = {"_parse_error": str(je), "_line": i}
                    records.append(obj)
                    if rows and len(records) >= rows:
                        break
            out["friday_preferences.jsonl"] = pd.DataFrame(records)
        except Exception as e:
            out["friday_preferences.jsonl"] = pd.DataFrame({"error": [str(e)], "file": [jsonl_path]})

    # Pools directory: load ALL CSVs recursively under TrainingData/pools
    if os.path.isdir(pools_base):
        for root, _, files in os.walk(pools_base):
            for fname in files:
                if fname.lower().endswith(".csv"):
                    path = os.path.join(root, fname)
                    key = os.path.relpath(path, training_base)
                    try:
                        df = pd.read_csv(path)
                        if rows and len(df) > rows:
                            df = df.head(rows)
                        out[key] = df
                    except Exception as e:
                        out[key] = pd.DataFrame({"error": [str(e)], "file": [path]})

    return out


def format_summary(datasets: Dict[str, pd.DataFrame], limit: int = 5) -> str:
    chunks = []
    for name, df in datasets.items():
        chunks.append(f"\n=== {name} ===")
        try:
            chunks.append(f"Rows: {len(df)} | Columns: {list(df.columns)}")
            chunks.append(df.head(limit).to_string(index=False))
        except Exception as e:
            chunks.append(f"Failed to render DataFrame: {e}")
    return "\n".join(chunks)


def save_csvs(datasets: Dict[str, pd.DataFrame], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, df in datasets.items():
        safe_name = name.replace("/", "_")
        path = os.path.join(out_dir, f"{safe_name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Load and inspect local TrainingData datasets")
    parser.add_argument("--rows", type=int, default=1000, help="Max rows to load per file")
    parser.add_argument("--limit", type=int, default=5, help="Rows to show per dataset head()")
    parser.add_argument("--save", type=str, default="", help="Optional output directory to save CSVs")
    parser.add_argument("--out-text", type=str, default="", help="Optional path to write printed summary text")
    args = parser.parse_args()

    datasets = load_training_data(rows=args.rows)
    summary = format_summary(datasets, limit=args.limit)

    if args.out_text:
        with open(args.out_text, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        print(f"Wrote summary to: {args.out_text}")
    else:
        print(summary)

    if args.save:
        save_csvs(datasets, args.save)


if __name__ == "__main__":
    main()
