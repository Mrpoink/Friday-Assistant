import argparse
from typing import Dict, List, Tuple

import os
import pandas as pd
from datasets import load_dataset


def _safe_to_pandas(hf_ds, take: int) -> pd.DataFrame:
    try:
        if take:
            hf_ds = hf_ds.select(range(min(take, len(hf_ds))))
        return hf_ds.to_pandas()
    except Exception:
        try:
            records = hf_ds.select(range(min(take, len(hf_ds)))).to_list()
            return pd.DataFrame(records)
        except Exception:
            return pd.DataFrame()


def load_base_datasets(rows: int) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    loaders: List[Tuple[str, dict]] = [
        ("hh-rlhf", {"path": "Anthropic/hh-rlhf", "kwargs": {"split": "train"}}),
        ("self-cognition", {"path": "modelscope/self-cognition", "kwargs": {"split": "train"}}),
        ("OpenThoughts-114k", {"path": "open-thoughts/OpenThoughts-114k", "kwargs": {"split": "train"}}),
        ("Bespoke-Stratos-17k", {"path": "bespokelabs/Bespoke-Stratos-17k", "kwargs": {"split": "train"}}),
        ("glaive-function-calling-v2", {"path": "glaiveai/glaive-function-calling-v2", "kwargs": {"split": "train"}}),
        ("Magicoder-Evol-Instruct-110K", {"path": "ise-uiuc/Magicoder-Evol-Instruct-110K", "kwargs": {"split": "train"}}),
        ("Infinity-Instruct-7M_core", {"path": "BAAI/Infinity-Instruct", "kwargs": {"name": "7M_core", "split": "train"}}),
        ("Open-Platypus", {"path": "garage-bAInd/Open-Platypus", "kwargs": {"split": "train"}}),
        ("empathetic_dialogues", {"path": "facebook/empathetic_dialogues", "kwargs": {"split": "train", "trust_remote_code": True}}),
    ]

    for name, spec in loaders:
        try:
            hf = load_dataset(spec["path"], **spec["kwargs"])  # type: ignore
            if isinstance(hf, dict):
                ds = hf.get("train") or next(iter(hf.values()))
            else:
                ds = hf
            df = _safe_to_pandas(ds, rows)
            out[name] = df
        except Exception as e:
            out[name] = pd.DataFrame({"error": [str(e)]})

    # Local SMS dataset if present
    tpath = os.path.join(os.getcwd(), "TrainingData", "training_data.csv")
    if os.path.isfile(tpath):
        try:
            df_local = pd.read_csv(tpath)
            if rows and len(df_local) > rows:
                df_local = df_local.head(rows)
            out["sms-local"] = df_local
        except Exception as e:
            out["sms-local"] = pd.DataFrame({"error": [str(e)]})

    return out


def format_summary(datasets: Dict[str, pd.DataFrame], limit: int = 1) -> str:
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
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Load and inspect base datasets (no Datsets)")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows per dataset to load")
    parser.add_argument("--limit", type=int, default=5, help="Rows to show per dataset head()")
    parser.add_argument("--save", type=str, default="", help="Optional output directory to save CSVs")
    parser.add_argument("--out-text", type=str, default="", help="Optional path to write printed summary text")
    args = parser.parse_args()

    datasets = load_base_datasets(rows=args.rows)
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
