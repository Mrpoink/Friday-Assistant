import os
import csv
import json
import time
import argparse
from datetime import datetime
import pandas as pd

# Reuse logic from the existing epoch sampler to ensure identical processing
from epoch_sampler import (
    build_pools,
    load_static_sms,
    load_identity_qa_pairs,
    sample_with_replacement,
    parse_pool_percentages,
    ollama_generate_think,
    EPOCH_DIR,
    STATIC_SMS_ROWS,
    STATIC_IDENTITY_QA_ROWS,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_csv(df: pd.DataFrame, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["messages","target","source"]) 
        writer.writeheader()
        for _, row in df.iterrows():
            writer.writerow({
                "messages": row.get("messages", "[]"),
                "target": row.get("target", ""),
                "source": row.get("source", "friday")
            })


def append_csv(df: pd.DataFrame, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    file_exists = os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["messages","target","source"]) 
        if not file_exists:
            writer.writeheader()
        for _, row in df.iterrows():
            writer.writerow({
                "messages": row.get("messages", "[]"),
                "target": row.get("target", ""),
                "source": row.get("source", "friday")
            })


def sample_pool_epoch(pools: dict[str, pd.DataFrame],
                      counts: dict[str, int],
                      static_sms: pd.DataFrame,
                      static_identity: pd.DataFrame,
                      pool_key: str) -> pd.DataFrame:
    """Sample a single pool for one epoch, including static anchors if requested."""
    dfs = []
    # Include static anchors alongside dynamic pool to preserve distribution
    if STATIC_SMS_ROWS > 0:
        dfs.append(static_sms.sample(n=min(STATIC_SMS_ROWS, len(static_sms)), replace=False, random_state=42))
    if STATIC_IDENTITY_QA_ROWS > 0:
        dfs.append(static_identity.sample(n=STATIC_IDENTITY_QA_ROWS, replace=True, random_state=42))
    # Dynamic pool portion
    n = counts.get(pool_key, 0)
    df_pool = pools.get(pool_key, pd.DataFrame(columns=['messages','target','source']))
    if len(df_pool) > 0 and n > 0:
        dfs.append(sample_with_replacement(df_pool, n))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['messages','target','source'])


def build_pool_epochs(args):
    ensure_dir(EPOCH_DIR)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Building per-pool epochs… epochs={args.epochs}")

    static_sms = load_static_sms()
    static_identity = load_identity_qa_pairs(STATIC_IDENTITY_QA_ROWS)
    pools = build_pools()
    counts = parse_pool_percentages(args)

    start_time = time.time()
    # Create root folders for cumulative pool outputs
    pools_root = os.path.join("TrainingData", "pools")
    ensure_dir(pools_root)

    for epoch in range(1, args.epochs + 1):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch {epoch} (per-pool)")
        loop_start = time.time()
        total_rows = 0

        for pool_key in counts.keys():
            # Sample data for this pool
            df = sample_pool_epoch(pools, counts, static_sms, static_identity, pool_key)

            # Optional Ollama augmentation per-pool, mirroring original faucet behavior
            if getattr(args, "ollama_thoughts", False) and len(df) > 0:
                def _pb_generic(row):
                    try:
                        ctx = json.loads(row.get("messages", "[]"))
                        # Use most recent user message when available
                        for m in reversed(ctx):
                            if m.get("role") == "user":
                                return f"Reason step-by-step about: {m.get('content','')}"
                        return f"Reason step-by-step about: {ctx[0].get('content','') if ctx else ''}"
                    except Exception:
                        return ""
                def _pb_tools(row):
                    try:
                        ctx = json.loads(row.get("messages", "[]"))
                        user_msg = " \n".join(m.get("content", "") for m in ctx if m.get("role") == "user")
                        return f"Plan tool usage: {user_msg}"
                    except Exception:
                        return ""
                def _pb_sms(row):
                    try:
                        ctx = json.loads(row.get("messages", "[]"))
                        for m in reversed(ctx):
                            if m.get("role") == "user":
                                return f"Reflect on prior SMS: {m.get('content','')}"
                        return ""
                    except Exception:
                        return ""

                # Choose prompt builder based on pool
                if pool_key in ("tools_glaive", "tools"):
                    prompt_builder = _pb_tools
                elif pool_key in ("intel_magicoder", "intel_reclor", "intel_openmix", "logic", "logic_text"):
                    prompt_builder = _pb_generic
                else:
                    prompt_builder = _pb_generic

                # Optionally restrict SMS augmentation to when flag is set
                if pool_key == "friday" and not getattr(args, "ollama_sms", False):
                    prompt_builder = None

                if prompt_builder is not None:
                    out_rows = []
                    for _, row in df.iterrows():
                        try:
                            current_target = row.get('target', '')
                            if "<think>" in str(current_target):
                                think = ""
                            else:
                                prompt = prompt_builder(row)
                                think = ollama_generate_think(prompt) if prompt else ""
                            if think:
                                row["target"] = f"{think}\n{current_target}".strip()
                        except Exception:
                            pass
                        out_rows.append(row)
                    df = pd.DataFrame(out_rows)

            # Write epoch-specific file and append to cumulative per-pool CSV
            out_epoch_dir = os.path.join(EPOCH_DIR, pool_key)
            ensure_dir(out_epoch_dir)
            out_epoch_path = os.path.join(out_epoch_dir, f"epoch_{epoch}.csv")
            to_csv(df, out_epoch_path)

            cumulative_path = os.path.join(pools_root, f"{pool_key}.csv")
            append_csv(df, cumulative_path)

            total_rows += len(df)
            try:
                by_source = df['source'].value_counts().to_dict()
            except Exception:
                by_source = {}
            think_count = df['target'].astype(str).str.contains('<think>').sum()
            tool_count = df['target'].astype(str).str.contains('<tool_call>').sum()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pool {pool_key}: wrote {len(df)} rows -> {out_epoch_path}; cumulative -> {cumulative_path}")
            print(f"Composition: {by_source} | Tags: <think>={think_count}, <tool_call>={tool_count}")

        loop_elapsed = time.time() - loop_start
        overall_elapsed = time.time() - start_time
        avg_per_epoch = overall_elapsed / epoch
        remaining = args.epochs - epoch
        eta_sec = int(avg_per_epoch * remaining)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed epoch {epoch} (rows={total_rows}) in {int(loop_elapsed)}s | ETA: {eta_sec}s")

    total_elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All per-pool epochs completed in {int(total_elapsed)}s (~{int(total_elapsed/60)} min)")


def make_argparser():
    p = argparse.ArgumentParser(description="Per-pool epoch sampler (incremental appends)")
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs to generate")
    p.add_argument("--ollama_thoughts", action="store_true", help="Augment datasets with Ollama-generated <think> thoughts per pool")
    p.add_argument("--ollama_sms", action="store_true", help="Also augment static SMS anchors with Ollama-generated thoughts (off by default)")
    # Pool-level percentages (0-100) reused from epoch sampler
    for k in [
        "identity_hh","self_ultra","think_openthoughts","rag_bespoke","tools_glaive",
        "intel_magicoder","intel_reclor","intel_openmix","create_dolphin","create_airoboros",
        "enigmata","empathetic"
    ]:
        p.add_argument(f"--{k}", type=int, default=100, help=f"Percentage for pool {k}")
    p.add_argument("--double_non_static", action="store_true", help="Double the size of all dynamic pools; static anchors remain fixed")
    return p


if __name__ == "__main__":
    args = make_argparser().parse_args()
    build_pool_epochs(args)
