import os
import json
import time
from datetime import datetime
import pandas as pd

from Download_datasets import Datsets
from ExtraTools import prepend_system
from datasets import load_dataset

# Output directory for per-pool CSVs
OUT_DIR = os.path.join("TrainingData", "pools")

# Fixed sizes per pool, matching epoch_sampler names
POOL_SIZES = {
    "identity_hh": 2880,
    "self_ultra": 3360,
    "think_openthoughts": 4320,
    "rag_bespoke": 3840,  # maps to bespoke_stratos
    "tools_glaive": 4800,  # maps to glaive_fc_v2
    "intel_magicoder": 5760,  # maps to magicoder_evol
    "intel_reclor": 4800,  # maps to reclor
    "intel_openmix": 3840,  # maps to infinity_instruct + open_platypus
    "create_dolphin": 2400,
    "create_airoboros": 2400,
    "enigmata": 2000,
    "empathetic": 2000,  # maps to empathetic_dialogues
}

# Optional per-pool percentage tweaks (0-100). No CLI.
POOL_PCT = {k: 100 for k in POOL_SIZES.keys()}

# Micro-batch size for parallel thinking
THINK_BATCH_SIZE = 16



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def progress_bar_inline(label: str, processed: int, total: int, start_time: float, width: int = 30):
    pct = 0 if total == 0 else processed / total
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = int(time.time() - start_time)
    remaining = max(0, total - processed)
    eta = int((elapsed / processed) * remaining) if processed > 0 else 0
    msg = f"{label}: [{bar}] {processed}/{total} ({int(pct*100)}%) | left {remaining} | elapsed {elapsed}s | ETA {eta}s"
    print("\r" + msg, end="", flush=True)


def _strip_identity_lines(text: str) -> str:
    """Remove identity tag lines like `[identity] server:self local machine` from any text."""
    try:
        lines = str(text or "").splitlines()
        cleaned = [ln for ln in lines if not ln.strip().lower().startswith("[identity] server:self local machine")]
        return "\n".join(cleaned)
    except Exception:
        return str(text or "")


def _sanitize_messages_identity(msgs_json: str) -> str:
    """Remove identity tag lines from each message content in the messages JSON payload."""
    try:
        msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
    except Exception:
        msgs = []
    if not isinstance(msgs, list):
        msgs = []
    for m in msgs:
        if isinstance(m, dict):
            m["content"] = _strip_identity_lines(m.get("content", ""))
    return json.dumps(msgs)


def _insert_assistant_think(msgs_json: str, think_text: str) -> str:
    """Insert an assistant message containing the think_text, avoiding duplicates."""
    try:
        msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
    except Exception:
        msgs = []
    if not isinstance(msgs, list):
        msgs = []
    # prevent duplicate think insertion
    for m in msgs:
        if m.get('role') == 'assistant' and '<think>' in str(m.get('content','')):
            return json.dumps(msgs)
    # insert before last assistant or append
    inserted = False
    for i in range(len(msgs)-1, -1, -1):
        if msgs[i].get('role') == 'assistant':
            msgs.insert(i, {"role":"assistant","content": think_text})
            inserted = True
            break
    if not inserted:
        msgs.append({"role":"assistant","content": think_text})
    return json.dumps(msgs)


def _extract_delta_and_emotion(msgs_json: str) -> tuple[str, str]:
    """Parse DELTA:(...) and first <emotion> token from the last user message content."""
    delta = None
    emotion = None
    try:
        msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
    except Exception:
        msgs = []
    if isinstance(msgs, list) and msgs:
        # find last user content
        for i in range(len(msgs)-1, -1, -1):
            if msgs[i].get('role') == 'user':
                content = str(msgs[i].get('content', ''))
                break
        else:
            content = ''
        import re as _re
        m = _re.search(r"\[DELTA:\(([^)]+)\)\]", content)
        if m:
            delta = m.group(1).strip()
        # emotion like <curiosity> or <gratitude>
        m2 = _re.search(r"<([a-zA-Z_]+)>", content)
        if m2:
            emotion = m2.group(1).strip()
    # defaults
    return (delta or "UNKNOWN", emotion or "neutral")


def _normalize_think_with_metadata(thought_text: str, delta: str, emotion: str) -> str:
    """Ensure a single well-formed <think>...</think> containing metadata and the original inner content.
    - Extract inner content between first <think> and </think>. If missing, use full text as inner.
    - Strip any stray </think> or <think> fragments from inner.
    - Build exactly one pair: <think>[delta: ...][emotion: ...] {inner}</think>
    """
    import re as _re
    t = str(thought_text or "")
    m = _re.search(r"<think>(.*?)</think>", t, flags=_re.DOTALL | _re.IGNORECASE)
    inner = m.group(1) if m else t
    # Remove any residual think tags inside
    inner = _re.sub(r"</?think>", "", inner)
    # Trim leading artifacts like leading quotes caused by truncation
    inner = inner.strip()
    # Repair leading truncation artifacts like "'s ..." or "’s ..."
    # Remove a leading possessive fragment if present: ^(['"’]s\s+)
    inner = _re.sub(r"^(['\"’]s\s+)", "", inner)
    # Basic capitalization fix if inner starts with lowercase alpha
    if inner[:1].islower():
        inner = inner[:1].upper() + inner[1:]
    return f"<think>[delta: {delta}][emotion: {emotion}] {inner}</think>"


    def to_csv(df: pd.DataFrame, out_path: str):
        ensure_dir(os.path.dirname(out_path))
        df = df[["messages", "target", "source"]].copy()
        df.to_csv(out_path, index=False, encoding="utf-8")


def build_pool_csvs():
    log("Initializing dataset builder…")
    ds = Datsets()

    # Map keys between epoch_sampler naming and Download_datasets outputs
    build_map = {
        "identity_hh": ds.return_hh,
        "self_ultra": ds.return_self_cog,
        "think_openthoughts": ds.return_openthoughts,
        "rag_bespoke": ds.return_bespoke,
        "tools_glaive": ds.return_glaive_fc,
        "intel_magicoder": ds.return_magicoder,
        # For ReClor, build a simple mapping inline
        "intel_reclor": None,
        "intel_openmix": None,  # handled by combining below
        "create_dolphin": None,
        "create_airoboros": None,
        "enigmata": None,
        "empathetic": ds.return_empathetic_dialogues,
    }

    written_summary = {}
    # Build individual pools
    for pool, size in POOL_SIZES.items():
        start = time.time()
        log(f"Building pool '{pool}' with target size {size} …")
        # Apply percentage tweak
        size = max(0, int(size * (POOL_PCT.get(pool, 100) / 100)))

        if pool == "intel_openmix":
            # Combine Infinity-Instruct and Open-Platypus half/half
            half = max(0, size // 2)
            df1 = ds.return_infinity(half)
            df2 = ds.return_open_platypus(size - half)
            df = pd.concat([df1, df2], ignore_index=True)
        elif pool == "intel_reclor":
            reclor = load_dataset("voidful/ReClor", data_files={"train": "train.json"}, split="train")
            avail = len(reclor)
            take = min(size, avail)
            reclor = reclor.select(range(take))
            def reclor_conv(ex):
                context = str(ex.get('context',''))
                question = str(ex.get('question',''))
                answers = ex.get('answers', [])
                label = ex.get('label')
                user_text = (context + "\n\nQuestion: " + question).strip()
                target = ''
                if isinstance(label, int) and isinstance(answers, list) and 0 <= label < len(answers):
                    target = answers[label]
                msgs = json.dumps([{ "role":"user","content": user_text }])
                return {"messages": prepend_system(msgs, ds.system_prompt), "target": target, "source": "logic_text"}
            df = pd.DataFrame(reclor.map(reclor_conv))
            if len(df) < size:
                df = df.sample(n=size, replace=True, random_state=42)
        elif pool == "create_dolphin":
            try:
                dolphin_ds = load_dataset(
                    "HuggingFaceEvalInternal/cognitivecomputations__dolphin-2.9-llama3-8b-details-private",
                    name="cognitivecomputations__dolphin-2.9-llama3-8b__leaderboard_arc_challenge",
                    split="latest"
                )
                avail = getattr(dolphin_ds, "num_rows", len(dolphin_ds))
                take = min(size, avail)
                dolphin_ds = dolphin_ds.select(range(take))
                def conv(example):
                    instr = example.get('instruction', example.get('prompt', example.get('question','')))
                    resp = example.get('output', example.get('response', example.get('answer','')))
                    msgs = json.dumps([{ "role":"user","content": str(instr or '') }])
                    return {"messages": prepend_system(msgs, ds.system_prompt), "target": str(resp or ''), "source": "dolphin_style"}
                df = pd.DataFrame(dolphin_ds.map(conv))
                if len(df) < size:
                    df = df.sample(n=size, replace=True, random_state=42)
            except Exception as e:
                log(f"Dolphin load failed: {e}")
                df = pd.DataFrame(columns=["messages","target","source"]) 
        elif pool == "create_airoboros":
            try:
                ds_stream = load_dataset("jondurbin/airoboros-2.2", split="train", streaming=True)
                rows = []
                for i, ex in enumerate(ds_stream):
                    if i >= size:
                        break
                    instr = ex.get('instruction', ex.get('prompt', ex.get('question', '')))
                    resp = ex.get('output', ex.get('response', ex.get('answer', '')))
                    msgs = json.dumps([{ "role":"user","content": str(instr or '') }])
                    rows.append({
                        "messages": prepend_system(msgs, ds.system_prompt),
                        "target": str(resp or ''),
                        "source": "airoboros_style"
                    })
                df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["messages","target","source"]) 
                if len(df) < size and len(df) > 0:
                    df = df.sample(n=size, replace=True, random_state=42)
            except Exception as e:
                log(f"Airoboros load failed: {e}")
                df = pd.DataFrame(columns=["messages","target","source"]) 
        elif pool == "enigmata":
            try:
                enigmata = load_dataset("BytedTsinghua-SIA/Enigmata-Data", split="train", trust_remote_code=True)
                avail = getattr(enigmata, "num_rows", len(enigmata))
                take = min(size, avail)
                enigmata = enigmata.select(range(take))
                def conv(example):
                    instr = example.get('instruction', example.get('prompt', example.get('question','')))
                    resp = example.get('output', example.get('response', example.get('answer','')))
                    msgs = json.dumps([{ "role":"user","content": str(instr or '') }])
                    return {"messages": prepend_system(msgs, ds.system_prompt), "target": str(resp or ''), "source": "enigmata"}
                df = pd.DataFrame(enigmata.map(conv))
                if len(df) < size:
                    df = df.sample(n=size, replace=True, random_state=42)
            except Exception as e:
                log(f"Enigmata load failed: {e}")
                df = pd.DataFrame(columns=["messages","target","source"]) 
        elif build_map.get(pool) is None:
            log(f"Pool '{pool}' not available via Download_datasets; writing empty CSV.")
            df = pd.DataFrame(columns=["messages","target","source"])  
        else:
            df = build_map[pool](size)

        # Strip identity tags in raw rows before any generation or insertion
        if len(df) > 0:
            df = df.copy()
            df["target"] = df["target"].apply(_strip_identity_lines)
            df["messages"] = df["messages"].apply(_sanitize_messages_identity)

        # Augment with model thoughts inline and show progress
        try:
            total = len(df)
        except Exception:
            total = 0
        processed = 0
        last_update = 0
        out_rows = []
        # use top-level _insert_assistant_think helper

        # Prepare batched generation for rows missing target <think>
        need_gen_indices = []
        need_gen_prompts = []
        extracted_thinks = {}
        for idx, row in df.iterrows():
            tgt = str(row.get("target", ""))
            msgs_json = row.get("messages", "")
            existing_think = None
            if "<think>" in tgt:
                import re as _re
                m = _re.search(r"<think>.*?</think>", tgt, flags=_re.DOTALL | _re.IGNORECASE)
                if m:
                    existing_think = m.group(0)
                    extracted_thinks[idx] = existing_think
            if not existing_think:
                user_prompt = ds._extract_last_user(msgs_json)
                # Condition thought on both user input and current target to learn how they relate
                combined = (
                    f"User message:\n{user_prompt}\n\nAssistant answer:\n{tgt}\n\n"
                    "Provide internal reasoning that connects the user's request to the assistant's answer."
                )
                need_gen_indices.append(idx)
                need_gen_prompts.append(combined)

        # Generate thoughts in micro-batches
        for i in range(0, len(need_gen_prompts), THINK_BATCH_SIZE):
            batch_prompts = need_gen_prompts[i:i+THINK_BATCH_SIZE]
            batch_indices = need_gen_indices[i:i+THINK_BATCH_SIZE]
            batch_thoughts = ds._generate_thinks(batch_prompts)
            for bi, bt in zip(batch_indices, batch_thoughts):
                extracted_thinks[bi] = bt
                processed += 1
                now = time.time()
                if now - last_update >= 0.1 or processed == total:
                    last_update = now
                    progress_bar_inline(f"Pool '{pool}'", processed, total, start)

        # Build output rows with inserted assistant <think>
        for idx, row in df.iterrows():
            msgs_json = row.get("messages", "")
            thought = extracted_thinks.get(idx)
            if thought is None:
                # fallback to existing think extracted from target or skip (shouldn't happen)
                thought = "<think>Internal reasoning.</think>"
            # Normalize with metadata to avoid duplication and cutoff
            try:
                delta, emotion = _extract_delta_and_emotion(msgs_json)
                thought = _normalize_think_with_metadata(thought, delta, emotion)
            except Exception:
                thought = _normalize_think_with_metadata(thought, "UNKNOWN", "neutral")
            row = row.copy()
            # Sanitize identity lines before inserting assistant think
            msgs_json_clean = _sanitize_messages_identity(msgs_json)
            row["messages"] = _insert_assistant_think(msgs_json_clean, thought)
            row["think_inserted"] = True
            # Ensure target also contains the thought at the beginning
            tgt = _strip_identity_lines(str(row.get("target", "")))
            if tgt:
                row["target"] = f"{thought}\n\n{tgt}"
            else:
                row["target"] = thought
            out_rows.append(row)
        # Finish line for progress bar
        print()
        df2 = pd.DataFrame(out_rows)

        out_path = os.path.join(OUT_DIR, f"{pool}.csv")
        df2.to_csv(out_path, index=False, encoding="utf-8")
        # Post-write validation: ensure every target starts with well-formed <think>…</think>
        try:
            audit = pd.read_csv(out_path)
            changed = False
            fixed_rows = []
            import re as _re
            for _, r in audit.iterrows():
                tgt = _strip_identity_lines(str(r.get("target", "")))
                msgs_json = _sanitize_messages_identity(r.get("messages", ""))
                # Extract first think block if present
                m = _re.search(r"<think>.*?</think>", tgt, flags=_re.DOTALL | _re.IGNORECASE)
                think = m.group(0) if m else None
                if not think:
                    # Try to recover from dangling tag fragments
                    clean = _re.sub(r"</?think>", "", tgt)
                    if clean.strip():
                        think = f"<think>{clean.strip()}</think>"
                    else:
                        # Generate a minimal reasoning based on both sides
                        user_prompt = ds._extract_last_user(msgs_json)
                        prompt = (
                            f"User message:\n{user_prompt}\n\nAssistant answer:\n{tgt}\n\n"
                            "Summarize the internal reasoning succinctly."
                        )
                        gen = ds._generate_think(prompt)
                        if "<think>" not in gen:
                            gen = f"<think>{gen.strip()}</think>"
                        think = gen
                    r = r.copy()
                    r["target"] = f"{think}\n\n" + _strip_identity_lines(str(r.get("target", "")))
                    r["messages"] = _insert_assistant_think(_sanitize_messages_identity(msgs_json), think)
                    r["think_inserted"] = True
                    changed = True
                fixed_rows.append(r)
            if changed:
                pd.DataFrame(fixed_rows)[["messages","target","source","think_inserted"]].to_csv(out_path, index=False, encoding="utf-8")
                log(f"Post-write audit fixed missing think blocks in {pool}.csv")
        except Exception as e:
            log(f"Post-write audit skipped for {pool}.csv: {e}")
        log(f"Wrote {pool}.csv with {len(df2)} rows in {int(time.time()-start)}s")
        written_summary[pool] = len(df2)


# Minimal stub to satisfy type in build_map without importing datasets at top-level here.
def load_dataset_stub(*args, **kwargs):
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


if __name__ == "__main__":
    # Build a small 10-row sample first so it's available immediately
    try:
        print("\nBuilding 10-row sample from 'identity_hh' → _sample_identity_hh.csv …")
        ds = Datsets()
        df = ds.return_hh(10)
        if len(df) > 0:
            df = df.copy()
            df["target"] = df["target"].apply(_strip_identity_lines)
            df["messages"] = df["messages"].apply(_sanitize_messages_identity)
        need_gen_prompts = []
        need_gen_indices = []
        extracted_thinks = {}
        for idx, row in df.iterrows():
            tgt = str(row.get("target",""))
            msgs_json = row.get("messages","")
            import re as _re
            m = _re.search(r"<think>.*?</think>", tgt, flags=_re.DOTALL | _re.IGNORECASE)
            if m:
                extracted_thinks[idx] = m.group(0)
            else:
                user_prompt = ds._extract_last_user(msgs_json)
                combined = (
                    f"User message:\n{user_prompt}\n\nAssistant answer:\n{tgt}\n\n"
                    "Provide internal reasoning that connects the user's request to the assistant's answer."
                )
                need_gen_indices.append(idx)
                need_gen_prompts.append(combined)
        if need_gen_prompts:
            thoughts = ds._generate_thinks(need_gen_prompts)
            for bi, bt in zip(need_gen_indices, thoughts):
                extracted_thinks[bi] = bt
        rows = []
        for idx, row in df.iterrows():
            thought = extracted_thinks.get(idx, "<think>Internal reasoning.</think>")
            try:
                delta, emotion = _extract_delta_and_emotion(row.get("messages",""))
                t_str = _normalize_think_with_metadata(thought, delta, emotion)
            except Exception:
                t_str = _normalize_think_with_metadata(thought, "UNKNOWN", "neutral")
            msgs_json_clean = _sanitize_messages_identity(row.get("messages",""))
            row = row.copy()
            row["messages"] = _insert_assistant_think(msgs_json_clean, t_str)
            tgt = _strip_identity_lines(str(row.get("target","")))
            row["target"] = f"{t_str}\n\n{tgt}" if tgt else t_str
            row["think_inserted"] = True
            rows.append(row)
        sample_df = pd.DataFrame(rows)
        out_path = os.path.join("TrainingData","pools","_sample_identity_hh.csv")
        sample_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Wrote _sample_identity_hh.csv with {len(sample_df)} rows")
    except Exception as e:
        print("Sample build failed:", e)

    # Build full pools
    build_pool_csvs()
    # After run, list pool CSVs and row counts
    try:
        base = os.path.join("TrainingData", "pools")
        files = sorted([f for f in os.listdir(base) if f.endswith('.csv')])
        print("\nSummary of written pool CSVs:")
        for f in files:
            path = os.path.join(base, f)
            try:
                df = pd.read_csv(path)
                print(f"- {f}: {len(df)} rows")
            except Exception as e:
                print(f"- {f}: error reading -> {e}")
    except Exception as e:
        print("Summary listing failed:", e)
