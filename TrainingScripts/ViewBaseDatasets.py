import argparse
import pandas as pd
from typing import Dict

from Download_datasets import Datsets


def load_all(num_rows: int, add_thoughts: bool = False) -> Dict[str, pd.DataFrame]:
    ds = Datsets()
    return ds.download_all(num_rows=num_rows, add_thoughts=add_thoughts)


def print_summary(datasets: Dict[str, pd.DataFrame], limit: int = 5) -> None:
    for name, df in datasets.items():
        print(f"\n=== {name} ===")
        print(f"Rows: {len(df)} | Columns: {list(df.columns)}")
        print(df.head(limit).to_string(index=False))


def save_csvs(datasets: Dict[str, pd.DataFrame], out_dir: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    for name, df in datasets.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Load and inspect base datasets")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows per dataset to load")
    parser.add_argument("--limit", type=int, default=5, help="Rows to show per dataset head()")
    parser.add_argument("--save", type=str, default="", help="Optional output directory to save CSVs")
    parser.add_argument("--add-thoughts", action="store_true", help="Augment with generated <think> blocks")
    args = parser.parse_args()

    datasets = load_all(num_rows=args.rows, add_thoughts=args.add_thoughts)
    print_summary(datasets, limit=args.limit)

    if args.save:
        save_csvs(datasets, args.save)


if __name__ == "__main__":
    main()
