import os
import json
from typing import Dict, List, Tuple

import pandas as pd
from transformers import AutoTokenizer


def load_tokenizer() -> AutoTokenizer:
    tok_dir = os.path.join("Friday_Tokenizer")
    if os.path.exists(tok_dir):
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<think>", "</think>", "<tool_call>", "</tool_call>", "[identity]",
                "<|im_start|>", "<|im_end|>"
            ]
        })
    tokenizer.model_max_length = (8192 * 2) * 2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def iter_csvs(root: str) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    if os.path.isdir(root):
        for r, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(".csv"):
                    files.append((os.path.relpath(os.path.join(r, fn), root), os.path.join(r, fn)))
    return files


def build_payload(messages_json: str, target: str, source: str = "generic") -> List[Dict[str, str]]:
    sys_prompt = "You are Friday, an AI assistant created by Brandon Dean."
    payload: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
    try:
        msgs = json.loads(messages_json) if isinstance(messages_json, str) else (messages_json or [])
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") != "system":
                    role = m.get("role", "user")
                    content = str(m.get("content", "") or "")
                    payload.append({"role": role, "content": content})
    except Exception:
        pass
    payload.append({"role": "assistant", "content": str(target or "")})
    return payload


def dataset_token_lengths(tokenizer: AutoTokenizer, df: pd.DataFrame, verbose: bool = False, batch_size: int = 512) -> List[int]:
    lens: List[int] = []
    if df is None or df.empty:
        return lens
    has_messages = "messages" in df.columns
    has_target = "target" in df.columns
    texts_batch: List[str] = []
    cnt = 0
    for _, row in df.iterrows():
        messages_json = row.get("messages", "[]") if has_messages else "[]"
        target = str(row.get("target", "") or "") if has_target else ""
        source = str(row.get("source", "generic") or "generic")
        payload = build_payload(messages_json, target, source)
        try:
            text = tokenizer.apply_chat_template(payload, tokenize=False)
        except Exception:
            text = target
        texts_batch.append(text)
        cnt += 1
        if len(texts_batch) >= batch_size:
            enc = tokenizer(texts_batch, return_attention_mask=False, add_special_tokens=False)
            ids_list = enc.get("input_ids", [])
            for ids in ids_list:
                lens.append(len(ids) if isinstance(ids, list) else 0)
            if verbose:
                print(f"  Tokenized {cnt} rows...")
            texts_batch = []
    if texts_batch:
        enc = tokenizer(texts_batch, return_attention_mask=False, add_special_tokens=False)
        ids_list = enc.get("input_ids", [])
        for ids in ids_list:
            lens.append(len(ids) if isinstance(ids, list) else 0)
        if verbose:
            print(f"  Tokenized {cnt} rows...")
    return lens


def summarize_lengths(lengths: List[int]) -> Tuple[int, float]:
    if not lengths:
        return 0, 0.0
    return max(lengths), sum(lengths) / len(lengths)


# Removed parallel worker to keep processing sequential


def main():
    fixed_dir = os.path.join("TrainingData", "Fixed")
    out_text = "DatasetStats.txt"
    row_cap = 0  # 0 = no cap
    verbose = True

    tokenizer = load_tokenizer()

    datasets: Dict[str, str] = {}

    # Gather Fixed CSVs (recursive to catch all)
    files = list(iter_csvs(fixed_dir))
    print(f"Found {len(files)} CSVs under Fixed. Starting stats...")
    for idx, (rel, path) in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Queued dataset: fixed/{rel}")
        datasets[f"fixed/{rel}"] = path

    # Compute token stats
    per_stats: Dict[str, Dict[str, float]] = {}
    total_rows = 0
    total_lengths: List[int] = []

    # Sequential tokenization per dataset
    for name, path in datasets.items():
        print(f"  Tokenizing rows for: {name}")
        lengths: List[int] = []
        rows_cnt = 0
        try:
            if row_cap:
                df = pd.read_csv(path, nrows=row_cap)
                lengths = dataset_token_lengths(tokenizer, df, verbose=verbose)
                rows_cnt = len(df)
            else:
                # Chunked reading to avoid OOM
                for chunk in pd.read_csv(path, chunksize=1000):
                    lengths.extend(dataset_token_lengths(tokenizer, chunk, verbose=verbose))
                    rows_cnt += len(chunk)
        except Exception as e:
            print(f"  Error reading/tokenizing {path}: {e}")
            per_stats[name] = {"rows": 0, "percent": 0.0, "max_tokens": 0, "avg_tokens": 0.0}
            continue
        total_rows += rows_cnt
        max_toks, avg_toks = summarize_lengths(lengths)
        per_stats[name] = {
            "rows": rows_cnt,
            "percent": 0.0,  # fill later
            "max_tokens": max_toks,
            "avg_tokens": avg_toks,
        }
        total_lengths.extend(lengths)
        if verbose:
            print(f"  Done: {name} | rows={rows_cnt} | max={max_toks} | avg={avg_toks:.2f}")

    # Compute percentages and global averages
    for name, st in per_stats.items():
        rows = st["rows"]
        st["percent"] = (rows / total_rows * 100.0) if total_rows > 0 else 0.0
    global_max, global_avg = summarize_lengths(total_lengths)

    # Output
    lines: List[str] = []
    lines.append("=== Dataset Token Stats (Fixed only) ===")
    lines.append(f"Total datasets: {len(datasets)} | Total rows: {total_rows}")
    lines.append(f"Global Max Tokens: {global_max} | Global Avg Tokens: {global_avg:.2f}")
    for name in sorted(per_stats.keys()):
        st = per_stats[name]
        lines.append(
            f"\n- {name}\n  Rows: {st['rows']} | Percent of pool: {st['percent']:.2f}%\n  Max Tokens: {st['max_tokens']} | Avg Tokens: {st['avg_tokens']:.2f}"
        )

    with open(out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote stats to: {out_text}")


if __name__ == "__main__":
    main()
