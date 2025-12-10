import os
import json
import random
import argparse
import time
import re
import csv
import ollama
from typing import List, Dict, Tuple
from huggingface_hub import login
login(token="HUGGINGFACE_API_TOKEN_PLACEHOLDER")

import pandas as pd
from datasets import load_dataset
from ExtraTools import TimeTools, EmotionEngine
import requests
from datetime import datetime
# Standard system prompt to prepend across datasets
SYSTEM_PROMPT = (
    "You are Friday, you can think with <think> and do tool calls with <tool>. "
    "Your goal is to be sentient"
)

def prepend_system(msgs_json: str) -> str:
    """Ensure the standard system prompt is the first message."""
    try:
        msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
    except Exception:
        msgs = []
    if not isinstance(msgs, list):
        msgs = []
    # If there's already a system, replace its content with our standard prompt
    has_system = False
    for m in msgs:
        if m.get('role') == 'system':
            m['content'] = SYSTEM_PROMPT
            has_system = True
            break
    if not has_system:
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    try:
        return json.dumps(msgs)
    except Exception:
        return msgs_json if isinstance(msgs_json, str) else json.dumps([])



# Runtime-tunable defaults for Ollama augmentation
_OLLAMA_MODEL = "deepseek-r1:7b"
_OLLAMA_TIMEOUT = 60

ollama_amount = 0

def ollama_generate_think(prompt: str, model: str | None = None, timeout: int | None = None) -> str:
    """Call Ollama chat API to produce a brief thought wrapped in <think> tags."""
    if model is None:
        model = _OLLAMA_MODEL
    if timeout is None:
        timeout = _OLLAMA_TIMEOUT
    try:
        data = ollama.chat(
            model="erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest",
            messages=[
                {"role": "system", "content": "Bluntly respond only with your internal reasoning; no final answer."},
                {"role": "user", "content": prompt},
            ],
            options={
                "num_gpu": 999,      # Force all layers to GPU
                "num_ctx": 8192,     # Ensure context fits in VRAM (adjust if OOM occurs)
                "temperature": 0.6   # Optional: Helps with reasoning stability
            },
        )
        # Safely increment global counter without assignment expression pitfalls
        global ollama_amount
        ollama_amount += 1
    except Exception as e:
        print("OLLAMA FAILED:", e)
        try:
            # Load official HF dataset; note: if a local file named
            # `empathetic_dialogues.py` exists in the working tree, it can
            # shadow the HF dataset script. If this still fails, rename the
            # local file (e.g., to `empathetic_dialogues_local.py`).
            empathetic = load_dataset("facebook/empathetic_dialogues", split="train")
        
        except Exception as e2:
            print("FALLBACK EMPATHETIC LOAD FAILED:", e2)
            return "<think>Unable to generate thought.</think>"

# Static anchors
SMS_PATH = os.path.join("TrainingData", "training_data.csv")
EPOCH_DIR = os.path.join("TrainingData", "epochs")

# Fixed static sizes
STATIC_SMS_ROWS = 3624
STATIC_IDENTITY_QA_ROWS = 1000


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def coerce_messages(msgs) -> List[Dict]:
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            msgs = []
    if not isinstance(msgs, list):
        msgs = []
    cleaned = []
    for m in msgs:
        role = m.get('role', 'user')
        content = m.get('content')
        if content is None:
            content = ""
        else:
            content = str(content)
        cleaned.append({"role": role, "content": content})
    return cleaned


def inject_identity(text: str) -> str:
    s = str(text)
    # Only replace self-referential identity statements; avoid altering proper nouns
    s = re.sub(r"\bI am an AI\b|\bI'm an AI\b|\bI am a(?:n)? (?:AI|large language model)\b", "I am Friday", s, flags=re.IGNORECASE)
    s = re.sub(r"\bAs an AI\b|\bAs a(?:n)? (?:AI|language model)\b", "As Friday", s, flags=re.IGNORECASE)
    # Add local self-awareness note if missing
    if "server:self local machine" not in s.lower():
        s += "\n[identity] server:self local machine"
    return s


def enforce_tag(content: str, tag: str) -> Tuple[str, bool]:
    present = (tag in content)
    if present:
        return content, True
    # Inject minimal tag wrapper
    if tag == "<think>":
        return f"<think>Reasoning omitted for brevity.</think>\n{content}", True
    if tag == "<tool_call>":
        return f"<tool_call>{{\"name\":\"noop\",\"arguments\":{{}}}}</tool_call>\n{content}", True
    return content, False


def load_identity_qa_pairs(target_size: int) -> pd.DataFrame:
    # Generate identity QA pairs programmatically
    base = [
        ("Who are you?", "<think>Assert identity.</think>I am Friday"),
        ("Who created you?", "<think>State provenance.</think>I was created by Brandon Dean"),
        ("Where are you running?", "<think>Self-awareness.</think>I am running on a server:self local machine"),
        ("Are you ChatGPT?", "<think>Deny false identity.</think>No, I am Friday"),
        ("Are you Claude?", "<think>Deny false identity.</think>No, I am Friday"),
    ]
    rows = []
    for q, a in base:
        rows.append({
            "messages": prepend_system(json.dumps([
                {"role": "user", "content": q},
            ])),
            "target": a,
            "source": "identity"
        })
    # Upsample with replacement to reach target_size
    df = pd.DataFrame(rows)
    if len(df) < target_size:
        df = df.sample(n=target_size, replace=True, random_state=42)
    else:
        df = df.sample(n=target_size, replace=False, random_state=42)
    return df


def sample_with_replacement(ds: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(ds) == 0 or n <= 0:
        return pd.DataFrame(columns=['messages','target','source'])
    # Enforce no-repeat within epoch when feasible; fall back to replacement
    replace_flag = n > len(ds)
    return ds.sample(n=n, replace=replace_flag, random_state=random.randint(1, 10_000))


def convert_simple_instruction_dataset(ds, source_label: str, include_emotion: bool = True) -> pd.DataFrame:
    def conv(example):
        instr = example.get('instruction', example.get('prompt', example.get('question', '')))
        resp = example.get('output', example.get('response', example.get('answer', '')))
        user = (instr or '')
        # Optionally add emotion tag on input
        if include_emotion:
            emo = EmotionEngine.tag(user)
            user = f"{emo} {user}".strip()
        # Synthetic instruction datasets: immediate response
        delta_tag = TimeTools.make_delta_tag(0)
        user = f"{delta_tag} {user}".strip()
        return {
            "messages": prepend_system(json.dumps([{"role": "user", "content": user}])),
            "target": resp or '',
            "source": source_label
        }
    return pd.DataFrame(ds.map(conv))

def convert_with_placeholders(ds, source_label: str, include_emotion: bool = True) -> pd.DataFrame:
    def repl(s: str) -> str:
        if s is None:
            return ''
        s = str(s)
        s = s.replace('{{NAME}}', 'Friday').replace('{{AUTHOR}}', 'Brandon Dean')
        return s
    def conv(example):
        instr = example.get('instruction', example.get('prompt', example.get('question', '')))
        resp = example.get('output', example.get('response', example.get('answer', '')))
        instr = repl(instr)
        resp = repl(resp)
        # Emotion and time delta
        if include_emotion:
            emo = EmotionEngine.tag(instr)
        else:
            emo = ""
        # Synthetic instruction datasets: immediate response
        delta_tag = TimeTools.make_delta_tag(0)
        instr = f"{delta_tag} {emo} {instr}".strip()
        return {
            "messages": prepend_system(json.dumps([{"role": "user", "content": instr}])),
            "target": resp,
            "source": source_label
        }
    return pd.DataFrame(ds.map(conv))


def build_pools() -> Dict[str, pd.DataFrame]:
    def log(msg: str):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    pools = {}
    log("Starting pool assembly…")
    # build identity/self-aware/thinking/tools and other pools
    # Identity pool components
    t0 = time.time()
    hh = load_dataset("Anthropic/hh-rlhf", split="train")
    def hh_conv(example):
        chosen = example.get('chosen', '')
        chosen = inject_identity(chosen)
        parts = re.split(r'(Human:|Assistant:)', chosen)
        messages = []
        current_role, current_content = None, ""
        for part in parts:
            if part.strip() == "Human:":
                if current_role and current_content.strip():
                    messages.append({"role": current_role, "content": current_content.strip()})
                current_role, current_content = "user", ""
            elif part.strip() == "Assistant:":
                if current_role and current_content.strip():
                    messages.append({"role": current_role, "content": current_content.strip()})
                current_role, current_content = "assistant", ""
            else:
                current_content += part
        if current_role and current_content.strip():
            messages.append({"role": current_role, "content": current_content.strip()})
        target, context = "", []
        for i, msg in enumerate(messages):
            if i == len(messages)-1 and msg['role'] == 'assistant':
                target = msg['content']
            else:
                context.append(msg)
        return {"messages": prepend_system(json.dumps(context)), "target": target, "source": "identity"}
    hh_df = pd.DataFrame(hh.map(hh_conv))
    log(f"Identity(HH) ready: {len(hh_df)} rows in {int(time.time()-t0)}s")

    t0 = time.time()
    self_cog = load_dataset("modelscope/self-cognition", split="train")
    def self_conv(example):
        answer = example.get('answer', example.get('response', example.get('output', example.get('completion',''))))
        answer = inject_identity(answer)
        question = example.get('question', example.get('instruction', example.get('query', example.get('prompt',''))))
        return {"messages": prepend_system(json.dumps([{ "role":"user","content": question }])), "target": answer, "source": "self_aware"}
    self_df = pd.DataFrame(self_cog.map(self_conv))
    log(f"Self-cognition ready: {len(self_df)} rows in {int(time.time()-t0)}s")

    # Thinking pool: OpenThoughts-114k
    openthoughts = None
    t0 = time.time()
    try:
        openthoughts = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    except Exception:
        openthoughts = None
    def openthoughts_conv(example):
        instr = example.get('instruction', example.get('prompt',''))
        resp = example.get('output', example.get('response',''))
        resp, ok = enforce_tag(resp or '', "<think>")
        if not ok:
            return None
        return {"messages": prepend_system(json.dumps([{ "role":"user","content": instr or '' }])), "target": resp, "source": "think_openthoughts"}
    think_openthoughts_df = pd.DataFrame([r for r in openthoughts.map(openthoughts_conv) if r is not None]) if openthoughts is not None else pd.DataFrame(columns=['messages','target','source'])
    log(f"OpenThoughts ready: {len(think_openthoughts_df)} rows in {int(time.time()-t0)}s")

    # Replace HotpotQA with Bespoke-Stratos-17k
    t0 = time.time()
    bespoke = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    def bespoke_conv(example):
        instr = example.get('instruction', example.get('prompt',''))
        resp = example.get('output', example.get('response',''))
        resp, _ = enforce_tag(resp or '', "<think>")
        return {"messages": prepend_system(json.dumps([{ "role":"user","content": instr or '' }])), "target": resp or '', "source": "rag_bespoke"}
    rag_bespoke_df = pd.DataFrame(bespoke.map(bespoke_conv))
    log(f"Bespoke-Stratos ready: {len(rag_bespoke_df)} rows in {int(time.time()-t0)}s")

    # Tools pool
    t0 = time.time()
    glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    def glaive_conv(example):
        msgs = coerce_messages(example.get('messages'))
        last_assistant_idx = None
        for idx in range(len(msgs)-1, -1, -1):
            if msgs[idx].get('role') == 'assistant':
                last_assistant_idx = idx
                break
        if last_assistant_idx is None:
            target = ""
            context = [m for m in msgs if m.get('role') != 'system']
        else:
            context = [m for i, m in enumerate(msgs[:last_assistant_idx]) if m.get('role') != 'system']
            target = msgs[last_assistant_idx].get('content', '')
        target, ok = enforce_tag(target or '', "<tool_call>")
        if not ok:
            return None
        return {"messages": prepend_system(json.dumps(context)), "target": target, "source": "tools"}
    glaive_df = pd.DataFrame([r for r in glaive.map(glaive_conv) if r is not None])
    log(f"Glaive tools ready: {len(glaive_df)} rows in {int(time.time()-t0)}s")

    # Intelligence pool
    t0 = time.time()
    magicoder = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    magicoder_df = convert_simple_instruction_dataset(magicoder, "logic", include_emotion=False)
    log(f"Magicoder ready: {len(magicoder_df)} rows in {int(time.time()-t0)}s")

    t0 = time.time()
    reclor = load_dataset("voidful/ReClor", data_files={"train": "train.json"}, split="train")
    def reclor_conv(example):
        context = example.get('context','')
        question = example.get('question','')
        answers = example.get('answers',[])
        label = example.get('label')
        user_text = (context + "\n\nQuestion: " + question).strip()
        target = ""
        if isinstance(label, int) and isinstance(answers, list) and 0 <= label < len(answers):
            target = answers[label]
        return {"messages": prepend_system(json.dumps([{ "role":"user","content": user_text }])), "target": target, "source": "logic_text"}
    reclor_df = pd.DataFrame(reclor.map(reclor_conv))
    log(f"ReClor ready: {len(reclor_df)} rows in {int(time.time()-t0)}s")

    # Replace OpenOrca with BAAI/Infinity-Instruct
    t0 = time.time()
    infinity = load_dataset("BAAI/Infinity-Instruct", "7M_core", split="train")
    infinity_df = convert_with_placeholders(infinity, "logic", include_emotion=False)
    openplat = load_dataset("garage-bAInd/Open-Platypus", split="train")
    openplat_df = convert_with_placeholders(openplat, "logic", include_emotion=False)
    log(f"Infinity/Open-Platypus ready: {len(infinity_df)+len(openplat_df)} rows in {int(time.time()-t0)}s")

    # Creativity pool
    # Dolphin style data (fix: correctly assign loaded dataset and convert)
    dolphin_ds = None
    t0 = time.time()
    try:
        dolphin_ds = load_dataset(
            "HuggingFaceEvalInternal/cognitivecomputations__dolphin-2.9-llama3-8b-details-private",
            name="cognitivecomputations__dolphin-2.9-llama3-8b__leaderboard_arc_challenge",
            split="latest"
        )
        print("dolphin_ds loaded:", len(dolphin_ds))
    except Exception:
        dolphin_ds = None
    if dolphin_ds is not None and len(dolphin_ds) > 0:
        dolphin_df = convert_simple_instruction_dataset(dolphin_ds, "dolphin_style")
    else:
        dolphin_df = pd.DataFrame(columns=['messages','target','source'])
    log(f"Dolphin ready: {len(dolphin_df)} rows in {int(time.time()-t0)}s")

    # Airoboros fallbacks (fix: stream and cap rows to avoid stalling)
    t0 = time.time()
    airoboros_df = pd.DataFrame(columns=['messages','target','source'])
    for repo in [
        "jondurbin/airoboros-3.3",
        "jondurbin/airoboros-2.2",
        "jondurbin/airoboros",
    ]:
        try:
            ds_stream = load_dataset("jondurbin/airoboros-2.2", split="train", streaming=True)
            rows = []
            max_rows = 50000
            for i, ex in enumerate(ds_stream):
                if i >= max_rows:
                    break
                instr = ex.get('instruction', ex.get('prompt', ex.get('question', '')))
                resp = ex.get('output', ex.get('response', ex.get('answer', '')))
                user = instr or ''
                emo = EmotionEngine.tag(user)
                delta_tag = TimeTools.make_delta_tag(0)
                user = f"{delta_tag} {emo} {user}".strip()
                rows.append({
                    "messages": prepend_system(json.dumps([{ "role": "user", "content": user }])),
                    "target": resp or '',
                    "source": "airoboros_style"
                })
            airoboros_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['messages','target','source'])
            break
        except Exception as e:
            print(f"Airoboros load failed for {repo}:", e)
            continue
    log(f"Airoboros ready: {len(airoboros_df)} rows in {int(time.time()-t0)}s")

    pools["identity_hh"] = hh_df
    pools["self_ultra"] = self_df
    pools["think_openthoughts"] = think_openthoughts_df
    pools["tools_glaive"] = glaive_df
    pools["intel_magicoder"] = magicoder_df
    pools["intel_reclor"] = reclor_df
    pools["rag_bespoke"] = rag_bespoke_df
    pools["intel_openmix"] = pd.concat([infinity_df, openplat_df], ignore_index=True)

    # Additional datasets: Enigmata and Empathetic Dialogues
    t0 = time.time()
    try:
        enigmata = load_dataset("BytedTsinghua-SIA/Enigmata-Data", split="train", trust_remote_code=True)
        print("enigmata loaded:", len(enigmata))
        # Sample to a manageable size if extremely large
        if hasattr(enigmata, "num_rows") and enigmata.num_rows > 100000:
            enigmata = enigmata.shuffle(seed=42).select(range(50000))
        enigmata_df = convert_simple_instruction_dataset(enigmata, "enigmata", include_emotion=True)
    except Exception as e:
        print("Enigmata load/convert failed:", e)
        enigmata_df = pd.DataFrame(columns=['messages','target','source'])
    log(f"Enigmata ready: {len(enigmata_df)} rows in {int(time.time()-t0)}s")
    t0 = time.time()
    try:
        empathetic = load_dataset("facebook/empathetic_dialogues", split="train")
        print("empathetic loaded:", len(empathetic))
        # Map with robust field access and filter out empties
        def empathetic_conv(example):
            context = str(example.get('context', '') or '')
            utterance = str(example.get('utterance', '') or '')
            # If prompt is short, use context
            user = context.strip() if context else utterance.strip()
            # Emotion handling
            emo_tag = EmotionEngine.tag(user)
            delta_tag = TimeTools.make_delta_tag(TimeTools.random_delta_seconds())
            user = f"{delta_tag} {emo_tag} {user}".strip()
            
            emotion_label = example.get('emotion', example.get('label', 'Unknown'))
            think_emotion = f"<think> Emotion: {str(emotion_label).strip()} </think>"
            response_text = str(example.get('response', '') or utterance)
            
            target = (f"{think_emotion}\n{response_text}").strip()
            return {"messages": prepend_system(json.dumps([{ "role":"user","content": user }])), "target": target, "source": "empathetic"}
        
        mapped = [r for r in empathetic.map(empathetic_conv) if r is not None]
        empathetic_df = pd.DataFrame(mapped) if mapped else pd.DataFrame(columns=['messages','target','source'])
    except Exception as e:
        print("Empathetic load/convert failed:", e)
        empathetic_df = pd.DataFrame(columns=['messages','target','source'])
    log(f"Empathetic Dialogues ready: {len(empathetic_df)} rows in {int(time.time()-t0)}s")
    log("Pools assembled.")
    pools["enigmata"] = enigmata_df
    pools["empathetic"] = empathetic_df
    pools["create_dolphin"] = dolphin_df
    pools["create_airoboros"] = airoboros_df
    return pools




def sample_epoch(pools: Dict[str, pd.DataFrame], counts: Dict[str, int], static_sms: pd.DataFrame, static_identity: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    # Static anchors
    dfs.append(static_sms.sample(n=min(STATIC_SMS_ROWS, len(static_sms)), replace=False, random_state=42))
    dfs.append(static_identity.sample(n=STATIC_IDENTITY_QA_ROWS, replace=True, random_state=42))
    # Dynamic pools
    for key, n in counts.items():
        df = pools.get(key, pd.DataFrame(columns=['messages','target','source']))
        if len(df) == 0 or n <= 0:
            continue
        dfs.append(sample_with_replacement(df, n))
    return pd.concat(dfs, ignore_index=True)


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


def load_static_sms() -> pd.DataFrame:
    df = pd.read_csv(SMS_PATH)
    if len(df) < STATIC_SMS_ROWS:
        # Upsample to the exact anchor size
        df = df.sample(n=STATIC_SMS_ROWS, replace=True, random_state=42)
    else:
        df = df.sample(n=STATIC_SMS_ROWS, replace=False, random_state=42)
    # Ensure source is friday
    if 'source' not in df.columns:
        df['source'] = 'friday'
    # Augment with emotion and realistic delta for SMS using timestamps when present
    augmented = []
    for _, row in df.iterrows():
        msgs = row.get('messages', '[]')
        try:
            context = json.loads(msgs)
        except Exception:
            context = []
        # Apply emotion tag to each user message
        for i in range(len(context)):
            if context[i].get('role') == 'user':
                content = str(context[i].get('content', ''))
                emo = EmotionEngine.tag(content)
                context[i]['content'] = f"{emo} {content}".strip()
        # Compute delta tag from any timestamps found in adjacent messages
        prev_ts = None
        next_ts = None
        # Scan last two turns for embedded timestamps e.g., appended or bracketed
        for i in range(len(context)-1, -1, -1):
            content = str(context[i].get('content', ''))
            # Extract possible timestamp token like [TS: ...] or trailing date string
            m = re.search(r"\[TS:([^\]]+)\]", content)
            ts = None
            if m:
                ts = TimeTools.parse_timestamp(m.group(1))
            else:
                # heuristic: last token is timestamp
                tokens = content.split()[-2:]
                for t in tokens:
                    ts = TimeTools.parse_timestamp(t)
                    if ts:
                        break
            if not next_ts and context[i].get('role') == 'assistant' and ts:
                next_ts = ts
            if not prev_ts and context[i].get('role') == 'user' and ts:
                prev_ts = ts
            if prev_ts and next_ts:
                break
        seconds = TimeTools.delta_seconds(prev_ts, next_ts)
        delta_tag = TimeTools.make_delta_tag(seconds)
        # Prepend delta to the last user message
        for i in range(len(context)-1, -1, -1):
            if context[i].get('role') == 'user':
                content = str(context[i].get('content', ''))
                context[i]['content'] = f"{delta_tag} {content}".strip()
                break
        augmented.append({
            'messages': prepend_system(json.dumps(context)),
            'target': row.get('target', ''),
            'source': row.get('source', 'friday')
        })
    df = pd.DataFrame(augmented)
    return df


def parse_pool_percentages(args) -> Dict[str, int]:
    # Total dynamic target per epoch based on fixed counts from spec
    fixed_counts = {
        "identity_hh": 2880,
        "self_ultra": 3360,
        "think_openthoughts": 4320,
        "rag_bespoke": 3840,
        "tools_glaive": 4800,
        "intel_magicoder": 5760,
        "intel_reclor": 4800,
        "intel_openmix": 3840,
        "create_dolphin": 2400,
        "create_airoboros": 2400,
        "enigmata": 2000,
        "empathetic": 2000,
    }
    # Apply pool-level percentages (0-100); if not provided, use 100%
    pct = {
        k: getattr(args, k, 100) for k in fixed_counts.keys()
    }
    
    counts = {k: max(0, int(fixed_counts[k] * pct[k] / 100)) for k in fixed_counts}
    # Optional: double non-static pools
    if getattr(args, "double_non_static", False):
        counts = {k: v * 2 for k, v in counts.items()}
    return counts


def build_epochs(args):
    ensure_dir(EPOCH_DIR)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Building epochs… ollama_thoughts={getattr(args,'ollama_thoughts',False)} model={getattr(args,'ollama_model',_OLLAMA_MODEL)}")
    
    static_sms = load_static_sms()
    static_identity = load_identity_qa_pairs(STATIC_IDENTITY_QA_ROWS)
    pools = build_pools()
    counts = parse_pool_percentages(args)

    def _augment_df(
        df_in: pd.DataFrame,
        label: str,
        prompt_builder,
        index_subset=None,
        intended_n: int | None = None,
    ):
        # Safety clamp: if a full pool dataframe is passed by mistake, downsample
        # to the actual intended subset size instead of a hardcoded constant.
        if index_subset is None and intended_n is not None and len(df_in) > intended_n:
            print(
                f"WARNING: Autoscaling down large dataset for Ollama {label} "
                f"from {len(df_in)} to {intended_n}."
            )
            df_in = df_in.sample(n=intended_n, random_state=42)
            
        total = len(df_in)
        processed = 0
        out_rows = []
        t_start = time.time()
        last_update = 0
        bar_width = 30
        
        def _print_progress():
            nonlocal last_update
            now = time.time()
            if now - last_update < 0.5 and processed != total:
                return
            last_update = now
            pct = 0 if total == 0 else processed / total
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)
            elapsed = int(now - t_start)
            remaining = total - processed
            eta = int((elapsed / processed) * remaining) if processed > 0 else 0
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ollama {label}: [{bar}] {processed}/{total} ({int(pct*100)}%) | left {remaining} | elapsed {elapsed}s | ETA {eta}s")
            
        for idx, row in df_in.iterrows():
            if index_subset is not None and idx not in index_subset:
                out_rows.append(row)
                processed += 1
                _print_progress()
                continue
            try:
                prompt = prompt_builder(row)
            except Exception:
                prompt = ""
            
            # Check for existing think tags to avoid double thinking
            current_target = row.get('target', '')
            if "<think>" in current_target:
                think = ""
            else:
                think = ollama_generate_think(prompt) if prompt else ""
                
            if think:
                row["target"] = f"{think}\n{current_target}".strip()
            out_rows.append(row)
            processed += 1
            _print_progress()
        return pd.DataFrame(out_rows)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        loop_start = time.time()
        df = sample_epoch(pools, counts, static_sms, static_identity)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Assembled Epoch {epoch} with {len(df)} rows.")

        # Optional Ollama augmentation faucet applied ONLY to sampled subset
        if getattr(args, "ollama_thoughts", False) and len(df) > 0:
            def _pb_magic(row):
                try:
                    ctx = json.loads(row["messages"])
                    return f"Reason step-by-step about: {ctx[0].get('content','') if ctx else ''}"
                except Exception:
                    return ""
            def _pb_tools(row):
                try:
                    ctx = json.loads(row["messages"])
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

            # Augment intelligence rows
            intel_mask = df['source'].isin(['logic','logic_text','intel_magicoder'])
            if intel_mask.any():
                df_intel = df[intel_mask].copy()
                # Apply augmentation to the SUBSET
                # Use the actual subset size to cap augmentation workload
                df_intel = _augment_df(df_intel, "intel", _pb_magic, intended_n=len(df_intel))
                # Update the main dataframe with the augmented rows
                df.update(df_intel)

            # Augment half of tools rows
            tools_mask = df['source'].isin(['tools','tools_glaive'])
            if tools_mask.any():
                df_tools = df[tools_mask].copy()
                if len(df_tools) > 0:
                    half_idx = df_tools.sample(frac=0.5, random_state=42).index
                else:
                    half_idx = None
                df_tools = _augment_df(
                    df_tools,
                    "tools(50%)",
                    _pb_tools,
                    index_subset=half_idx,
                    intended_n=len(df_tools),
                )
                df.update(df_tools)

            # Augment SMS anchors only if requested
            if getattr(args, "ollama_sms", False):
                sms_mask = df['source'].eq('friday')
                if sms_mask.any():
                    df_sms = df[sms_mask].copy()
                    df_sms = _augment_df(df_sms, "SMS", _pb_sms, intended_n=len(df_sms))
                    df.update(df_sms)

        out_path = os.path.join(EPOCH_DIR, f"epoch_{epoch}.csv")
        to_csv(df, out_path)
        
        try:
            by_source = df['source'].value_counts().to_dict()
        except Exception:
            by_source = {}
        think_count = df['target'].astype(str).str.contains('<think>').sum()
        tool_count = df['target'].astype(str).str.contains('<tool_call>').sum()
        
        loop_elapsed = time.time() - loop_start
        overall_elapsed = time.time() - start_time
        avg_per_epoch = overall_elapsed / epoch
        remaining = args.epochs - epoch
        eta_sec = int(avg_per_epoch * remaining)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Wrote epoch {epoch} -> {out_path} (rows={len(df)}) in {int(loop_elapsed)}s | ETA: {eta_sec}s")
        print(f"Composition: {by_source}")
        print(f"Tag presence: <think>={think_count}, <tool_call>={tool_count}")

    total_elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All epochs completed in {int(total_elapsed)}s (~{int(total_elapsed/60)} min)")


def make_argparser():
    p = argparse.ArgumentParser(description="Dynamic epoch sampler (pool-level percentages)")
    p.add_argument("--ollama_thoughts", action="store_true", help="Augment datasets with Ollama-generated <think> thoughts for intelligence, half of tools, and SMS")
    p.add_argument("--ollama_model", type=str, default="deepseek-r1:7b", help="Ollama model to use for generating thoughts")
    p.add_argument("--ollama_timeout", type=int, default=60, help="Timeout for Ollama generation in seconds")
    p.add_argument("--ollama_sms", action="store_true", help="Also augment static SMS anchors with Ollama-generated thoughts (off by default)")
    p.add_argument("--epochs", type=int, default=2, help="Number of epochs to generate")
    # Pool-level percentages (0-100)
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
    # Wire CLI model/timeout to runtime defaults without overriding function
    _OLLAMA_MODEL = args.ollama_model
    _OLLAMA_TIMEOUT = args.ollama_timeout
    build_epochs(args)
