import os
import json
import time
from datetime import datetime
import pandas as pd
import concurrent.futures  # Ensure this is imported

from Download_datasets import Datsets
from ExtraTools import prepend_system, inject_identity
from datasets import load_dataset

# Output directory for per-pool CSVs
OUT_DIR = os.path.join("TrainingData", "pools")

# Fixed sizes per pool, matching epoch_sampler names
POOL_SIZES = {
    #"identity_hh": 2880,
    # self_ultra disabled intentionally
    "think_openthoughts": 4320,
    #"rag_bespoke": 3840,  # maps to bespoke_stratos
    #"tools_glaive": 4800,  # maps to glaive_fc_v2
    #"intel_magicoder": 5760,  # maps to magicoder_evol
    #"intel_reclor": 4800,  # maps to reclor
    "intel_openmix": 7680,  # doubled; maps to infinity_instruct + open_platypus
    # "create_dolphin": 2400,  # commented out per request
    #"create_airoboros": 2400,
    #"enigmata": 2000,
    #"sms": 4500,  # SMS conversations
    #"empathetic": 2000,  # maps to empathetic_dialogues
}

# Optional per-pool percentage tweaks (0-100). No CLI.
POOL_PCT = {k: 100 for k in POOL_SIZES.keys()}

# Multiplier to scale per-pool target sizes ("epoch size" for builder)
POOL_SIZE_MULTIPLIER = 6

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def progress_bar_inline(label: str, processed: int, total: int, start_time: float, width: int = 30):
    pct = 0 if total == 0 else processed / max(1, total)
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    now = time.time()
    elapsed = max(1, int(now - start_time))
    remaining = max(0, total - processed)
    avg_per_item = (now - start_time) / max(1, processed)
    eta = int(avg_per_item * remaining) if processed > 0 else 0
    msg = f"{label}: [{bar}] {processed}/{total} ({int(pct*100)}%) | left {remaining} | elapsed {elapsed}s | ETA {eta}s"
    print("\r" + msg, end="", flush=True)

def _strip_identity_lines(text: str) -> str:
    try:
        lines = str(text or "").splitlines()
        cleaned = [ln for ln in lines if not ln.strip().lower().startswith("[identity] server:self local machine")]
        return "\n".join(cleaned)
    except Exception:
        return str(text or "")

def _sanitize_messages_identity(msgs_json: str) -> str:
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

def _consolidate_assistant_turn(msgs_json: str, think_text: str, final_reply: str | None = None) -> str:
    try:
        msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
    except Exception:
        msgs = []
    if not isinstance(msgs, list):
        msgs = []
    merged_content = str(think_text or "").strip()
    if final_reply and str(final_reply).strip():
        merged_content = f"{merged_content}\n\n{str(final_reply).strip()}"
    new_msgs = list(msgs)
    new_msgs.append({"role": "assistant", "content": merged_content})
    
    try:
        return json.dumps(new_msgs)
    except Exception:
        return msgs_json if isinstance(msgs_json, str) else json.dumps([])

def _extract_delta_and_emotion(msgs_json: str) -> tuple[str, str]:
    delta = None
    emotion = None
    try:
        msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
    except Exception:
        msgs = []
    if isinstance(msgs, list) and msgs:
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
        m2 = _re.search(r"<([a-zA-Z_]+)>", content)
        if m2:
            emotion = m2.group(1).strip()
    return (delta or "UNKNOWN", emotion or "neutral")

def _normalize_think_with_metadata(thought_text: str, delta: str, emotion: str) -> str:
    import re as _re
    t = str(thought_text or "")
    m = _re.search(r"<think>(.*?)</think>", t, flags=_re.DOTALL | _re.IGNORECASE)
    inner = m.group(1) if m else t
    inner = _re.sub(r"</?think>", "", inner)
    inner = inner.strip()
    inner = _re.sub(r"^(['\"’]s\s+)", "", inner)
    if inner[:1].islower():
        inner = inner[:1].upper() + inner[1:]
    return f"<think>[delta: {delta}][emotion: {emotion}] {inner}</think>"

def build_pool_csvs():
    log("Initializing dataset builder…")
    ds = Datsets()
    
    # === NEW: Set parallelism for Gemini ===
    PARALLEL_WORKERS = 25  # Run 25 requests at once!

    build_map = {
        #"identity_hh": ds.return_hh,
        #"rag_bespoke": ds.return_bespoke,
        #"tools_glaive": ds.return_glaive_fc,
        #"intel_magicoder": ds.return_magicoder,
        #"intel_reclor": None,
        # intel_openmix is constructed below from Infinity + Open-Platypus
        "intel_openmix": None,
        #"create_dolphin": None,
        #"create_airoboros": None,
        #"enigmata": None,
        #"sms": ds.return_sms,
        #"empathetic": ds.return_empathetic_dialogues,
    }

    # Worker function for parallel execution
    def _worker_gen(args):
        idx, msgs = args
        try:
            # Generate 1 thought
            bt = ds._generate_thinks_from_messages([msgs])[0]
            return idx, bt
        except Exception as e:
            return idx, f"<think>Internal reasoning unavailable: {e}</think>"

    for pool, size in POOL_SIZES.items():
        start = time.time()
        log(f"Building pool '{pool}' with target size {size} (x{POOL_SIZE_MULTIPLIER}) …")
        size = max(0, int(size * (POOL_PCT.get(pool, 100) / 100)))
        size *= POOL_SIZE_MULTIPLIER

        # [DATA LOADING LOGIC SAME AS BEFORE]
        if pool == "intel_openmix":
            # Compose from Infinity + Open-Platypus with half each
            half = max(0, size // 2)
            df1 = ds.return_infinity(half)
            df2 = ds.return_open_platypus(size - half)
            df = pd.concat([df1, df2], ignore_index=True)
            # Conform to intel_openmix template columns expected by RLAIF
            tmpl_cols = [
                "id","conversations","label","langdetect","source","reward",
                "messages","target","input","output","instruction","data_source"
            ]
            rows = []
            for i, r in df.iterrows():
                msgs_json = str(r.get("messages", ""))
                tgt = str(r.get("target", ""))
                src = str(r.get("source", "intel_openmix"))
                # Attempt to extract first user instruction for input/instruction
                try:
                    msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
                except Exception:
                    msgs = []
                instr = ""
                if isinstance(msgs, list):
                    for m in msgs:
                        if m.get("role") == "user":
                            instr = str(m.get("content", ""))
                            break
                rows.append({
                    "id": i,
                    "conversations": msgs_json,
                    "label": "",
                    "langdetect": "en",
                    "source": src,
                    "reward": "",
                    "messages": msgs_json,
                    "target": tgt,
                    "input": instr,
                    "output": tgt,
                    "instruction": instr,
                    "data_source": "intel_openmix"
                })
            df = pd.DataFrame(rows, columns=tmpl_cols)
            log(f"Built intel_openmix source rows {len(df)} (Infinity + Platypus); proceeding to RLAIF augmentation")
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
        # elif pool == "create_dolphin":
        #     log("create_dolphin disabled; skipping this pool as requested")
        #     df = pd.DataFrame(columns=["messages","target","source"]) 
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
                # Use streaming to avoid strict column consistency across multiple configs
                ds_stream = load_dataset("BytedTsinghua-SIA/Enigmata-Data", split="train", trust_remote_code=True, streaming=True)
                rows = []
                for i, ex in enumerate(ds_stream):
                    if i >= size:
                        break
                    instr = ex.get('instruction', ex.get('prompt', ex.get('question','')))
                    resp = ex.get('output', ex.get('response', ex.get('answer','')))
                    msgs = json.dumps([{ "role":"user","content": str(instr or '') }])
                    rows.append({
                        "messages": prepend_system(msgs, ds.system_prompt),
                        "target": str(resp or ''),
                        "source": "enigmata"
                    })
                df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["messages","target","source"]) 
                if len(df) < size and len(df) > 0:
                    df = df.sample(n=size, replace=True, random_state=42)
            except Exception as e:
                log(f"Enigmata load failed: {e}")
                df = pd.DataFrame(columns=["messages","target","source"]) 
        elif build_map.get(pool) is None:
            log(f"Pool '{pool}' not available via Download_datasets; writing empty CSV.")
            df = pd.DataFrame(columns=["messages","target","source"])  
        else:
            df = build_map[pool](size)

        if len(df) > 0:
            df = df.copy()
            df["target"] = df["target"].apply(_strip_identity_lines)
            df["messages"] = df["messages"].apply(_sanitize_messages_identity)

        try:
                total_rows = len(df)
        except Exception:
                total_rows = 0
        processed = 0
        last_update = 0
        out_rows = []

        need_gen_indices = []
        need_gen_messages = []
        extracted_thinks = {}
        
        # 1. Identify rows needing thoughts
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
                try:
                    msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
                except Exception:
                    msgs = []
                if not isinstance(msgs, list):
                    msgs = []
                if str(tgt).strip():
                    msgs.append({"role": "assistant", "content": str(tgt)})
                need_gen_indices.append(idx)
                need_gen_messages.append(msgs)

        # 2. PARALLEL GENERATION (Replaced sequential loop)
        if need_gen_indices:
            log(f"Generating thoughts for {len(need_gen_indices)} items (Parallel x{PARALLEL_WORKERS})...")
            work_items = list(zip(need_gen_indices, need_gen_messages))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                # Submit all jobs
                future_to_idx = {executor.submit(_worker_gen, item): item[0] for item in work_items}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx, bt = future.result()
                    extracted_thinks[idx] = bt
                    
                    processed += 1
                    if processed % 500 == 0:
                        
                        # We can't easily save the full CSV here without rebuilding the DF, 
                        # but we can save the dictionary of thinks.
                        backup_path = os.path.join(OUT_DIR, f"{pool}_backup.json")
                        with open(backup_path, "w") as f:
                            json.dump(extracted_thinks, f)
                            
                    now = time.time()
                    if now - last_update >= 0.5 or processed == len(work_items) + (total_rows - len(work_items)):
                        last_update = now
                        progress_bar_inline(f"Pool '{pool}'", processed + (total_rows - len(work_items)), total_rows, start)

        # 3. Build Final Rows
        for idx, row in df.iterrows():
            msgs_json = row.get("messages", "")
            thought = extracted_thinks.get(idx)
            if thought is None:
                thought = "<think>Internal reasoning.</think>"
            try:
                delta, emotion = _extract_delta_and_emotion(msgs_json)
                thought = _normalize_think_with_metadata(thought, delta, emotion)
            except Exception:
                thought = _normalize_think_with_metadata(thought, "UNKNOWN", "neutral")
            row = row.copy()
            msgs_json_clean = _sanitize_messages_identity(msgs_json)
            visible = _strip_identity_lines(str(row.get("target", "")))
            row["messages"] = _consolidate_assistant_turn(msgs_json_clean, thought, visible)
            row["think_inserted"] = True
            tgt = _strip_identity_lines(str(row.get("target", "")))
            if tgt:
                row["target"] = f"{thought}\n\n{tgt}"
            else:
                row["target"] = thought
            out_rows.append(row)
        
        print() # Newline after progress bar
        df2 = pd.DataFrame(out_rows)
        out_path = os.path.join(OUT_DIR, f"{pool}.csv")
        df2.to_csv(out_path, index=False, encoding="utf-8")
        duration = int(time.time()-start)
        log(f"Wrote {pool}.csv with {len(df2)} rows in {duration}s (source rows={total_rows}, generated={len(out_rows)})")

if __name__ == "__main__":
    build_pool_csvs()