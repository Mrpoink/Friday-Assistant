import argparse
import os
import re
import random
import json
from typing import List, Dict, Any, Tuple

import pandas as pd
from datasets import load_dataset
from ExtraTools import TimeTools, EmotionEngine
from tqdm.auto import tqdm
from transformers import AutoTokenizer

DELTA_TAG_PATTERN = re.compile(r"\[DELTA:\s*\(([^)]*)\)\]", re.IGNORECASE)
THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

DELTA_CHOICES = ["SHORT", "MEDIUM", "LONG", "EXTRA LONG", "FOREVER"]
def _wrap_im(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    # Avoid double-wrapping
    if text.startswith("<|im_start|>") and text.endswith("<|im_end|>"):
        return text
    return f"<|im_start|>{text}<|im_end|>"


def _normalize_special_tokens(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    s = text
    # Replace pipe-enclosed tokens like |think| or ||tool|| etc.
    s = re.sub(r"\|(think|thought|thinking)\|", "<think>", s, flags=re.IGNORECASE)
    s = re.sub(r"\|(tool[_\s-]?call|tool)\|", "<tool_call>", s, flags=re.IGNORECASE)

    # Convert angle brackets with known keywords into our tags
    # e.g., <think> or <tool> or <tool_call>
    s = re.sub(r"<\s*(think|thought|thinking)\s*>", "<think>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*(think|thought|thinking)\s*>", "</think>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*(tool[_\s-]?call|tool)\s*>", "<tool_call>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*(tool[_\s-]?call|tool)\s*>", "</tool_call>", s, flags=re.IGNORECASE)

    # For unknown tokens inside <> or ||, leave them as-is to avoid destructive changes
    return s


def _fix_delta(text: str) -> str:
    if not text:
        return text
    def repl(m: re.Match) -> str:
        val = (m.group(1) or "").strip().upper()
        if val == "UNKNOWN" or not val:
            return f"[DELTA:({random.choice(DELTA_CHOICES)})]"
        return m.group(0)
    return DELTA_TAG_PATTERN.sub(repl, text)


def _replace_ai_in_think(think_text: str) -> str:
    if not think_text:
        return think_text
    # Replace specific phrasings to persona language
    think_text = re.sub(r"(?i)\bAs an AI\b", "Because I am me", think_text)
    think_text = re.sub(r"(?i)\bAs a Large Language Model\b", "Because I am me", think_text)
    think_text = re.sub(r"(?i)\bAs a large language model\b", "Because I am me", think_text)
    think_text = re.sub(r"(?i)\bas an artificial intelligence\b", "because I am me", think_text)
    think_text = re.sub(r"(?i)\bas an AI language model\b", "because I am me", think_text)
    think_text = re.sub(r"(?i)\bas an AI model\b", "because I am me", think_text)
    think_text = re.sub(r"(?i)\bas an AI assistant\b", "because I am me", think_text)
    think_text = re.sub(r"(?i)\bas an ai\b", "because I am me", think_text)
    think_text = re.sub(r"(?i)\bas a language model\b", "because I am me", think_text)
    think_text = re.sub(r"(?i)\bas an artificial intelligence model\b", "because I am me", think_text)
    think_text = re.sub(r"(?i)\bAI language model\b", "me", think_text)
    think_text = re.sub(r"(?i)\bartificial intelligence\b", "me", think_text)
    # Replace standalone "AI" -> "me"
    think_text = re.sub(r"(?i)\bAI\b", "me", think_text)
    return think_text


def _fix_think_blocks(text: str) -> str:
    if not text or "<think>" not in text:
        return text
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        fixed = _replace_ai_in_think(inner)
        return f"<think>{fixed}</think>"
    return THINK_BLOCK_PATTERN.sub(repl, text)


def _process_text_field(val: Any) -> Any:
    if not isinstance(val, str):
        return val
    out = _fix_delta(val)
    out = _fix_think_blocks(out)
    out = _normalize_special_tokens(out)
    return out


def _parse_msgs(val: Any) -> list:
    if isinstance(val, str) and val.strip():
        try:
            obj = json.loads(val)
            if isinstance(obj, list):
                return obj
        except Exception:
            return []
    elif isinstance(val, list):
        return val
    return []


def _normalize_msgs(msgs: list) -> list:
    out = []
    for m in msgs or []:
        if isinstance(m, dict):
            role = m.get("role")
            content = _process_text_field(m.get("content", ""))
            if role in ("user", "assistant", "system"):
                # Ensure assistant outputs are wrapped with im tags
                if role == "assistant" and content:
                    content = _wrap_im(content)
                out.append({"role": role, "content": content})
    return out


def _ensure_target_in_messages(msgs_val: Any, target_val: Any) -> str:
    msgs = _parse_msgs(msgs_val)
    msgs = _normalize_msgs(msgs)
    t = target_val if isinstance(target_val, str) else ""
    t = _process_text_field(t)
    if t:
        need_append = True
        if msgs:
            last = msgs[-1]
            if isinstance(last, dict) and last.get("role") == "assistant" and str(last.get("content", "")) == t:
                need_append = False
        if need_append:
            msgs.append({"role": "assistant", "content": _wrap_im(t)})
    return json.dumps(msgs)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()

    # Determine conversation column
    conv_col = "messages" if "messages" in df.columns else ("conversations" if "conversations" in df.columns else None)

    def _row_proc(row: pd.Series) -> pd.Series:
        # Process common text-bearing columns
        for c in ("instruction", "prompt", "response", "output", "completion", "chosen", "rejected", "target"):
            if c in row:
                row[c] = _process_text_field(row[c])
        # Ensure target is appended to conversation
        if conv_col:
            row[conv_col] = _ensure_target_in_messages(row.get(conv_col, []), row.get("target", ""))
        else:
            # Create messages if missing and target exists
            t = row.get("target", "")
            t = t if isinstance(t, str) else ""
            t = _process_text_field(t)
            if t:
                row["messages"] = json.dumps([{ "role": "assistant", "content": _wrap_im(t) }])
        return row

    return df.apply(_row_proc, axis=1)


# ==== Tokenizer utilities and token-limit filtering ====
def _load_tokenizer() -> AutoTokenizer:
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
    tokenizer.model_max_length = 8192
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _collect_row_special_tokens(messages_str: str) -> List[str]:
    specials: List[str] = []
    # Collect DELTA tags
    for m in DELTA_TAG_PATTERN.finditer(messages_str or ""):
        tag = m.group(0)
        if tag not in specials:
            specials.append(tag)
    # Collect emotion tags heuristically (e.g., tokens like <emotion:...> or [EMOTION:...])
    emotion_matches = re.findall(r"<(?:emotion|affect):[^>]+>|\[EMOTION:[^\]]+\]", messages_str or "", flags=re.IGNORECASE)
    for em in emotion_matches:
        if em not in specials:
            specials.append(em)
    return specials


def _messages_to_text(tokenizer: AutoTokenizer, messages_json_str: str) -> str:
    try:
        msgs = json.loads(messages_json_str) if messages_json_str else []
        if not isinstance(msgs, list):
            msgs = []
    except Exception:
        msgs = []
    # Ensure minimal system if missing
    if not msgs or msgs[0].get("role") != "system":
        sys_prompt = "You are Friday, an AI assistant created by Brandon Dean."
        msgs = [{"role": "system", "content": sys_prompt}] + [m for m in msgs if isinstance(m, dict)]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False)
    except Exception:
        # Fallback: concatenate contents
        return "\n".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))


def filter_by_token_limit(df: pd.DataFrame, tokenizer: AutoTokenizer, limit: int = 8192) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    keep_idxs: List[int] = []
    # Pre-collect observed special tokens across df to add to tokenizer
    observed_specials: List[str] = []
    for i, row in df.iterrows():
        msgs_str = str(row.get("messages", ""))
        for tok in _collect_row_special_tokens(msgs_str):
            if tok not in observed_specials:
                observed_specials.append(tok)
    if observed_specials:
        tokenizer.add_special_tokens({"additional_special_tokens": observed_specials})

    for i, row in df.iterrows():
        msgs_str = str(row.get("messages", ""))
        text = _messages_to_text(tokenizer, msgs_str)
        try:
            ids = tokenizer(text, return_attention_mask=False, add_special_tokens=False)["input_ids"]
            length = len(ids)
        except Exception:
            length = 0
        if length <= limit:
            keep_idxs.append(i)
    if keep_idxs:
        return df.loc[keep_idxs].reset_index(drop=True)
    return df.head(0)


def verify_token_limit(df: pd.DataFrame, tokenizer: AutoTokenizer, limit: int = 8192) -> Tuple[pd.DataFrame, int]:
    """Recompute token lengths and drop any rows still exceeding the limit.
    Returns (filtered_df, dropped_count)."""
    if df is None or df.empty:
        return df, 0
    keep_idxs: List[int] = []
    for i, row in df.iterrows():
        msgs_str = str(row.get("messages", ""))
        text = _messages_to_text(tokenizer, msgs_str)
        try:
            ids = tokenizer(text, return_attention_mask=False, add_special_tokens=False)["input_ids"]
            length = len(ids)
        except Exception:
            length = 0
        if length <= limit:
            keep_idxs.append(i)
    dropped = len(df) - len(keep_idxs)
    if keep_idxs:
        return df.loc[keep_idxs].reset_index(drop=True), dropped
    return df.head(0), len(df)


# ==== Base dataset augmentation for OpenThoughts + Bespoke ====
BEGIN_THINK = re.compile(r"<\|begin_of_thought\|>", re.IGNORECASE)
END_THINK = re.compile(r"<\|end_of_thought\|>", re.IGNORECASE)
BEGIN_SOL = re.compile(r"<\|begin_of_solution\|>", re.IGNORECASE)
END_SOL = re.compile(r"<\|end_of_solution\|>", re.IGNORECASE)


def _extract_thought_and_solution(text: str) -> Tuple[str, str]:
    if not isinstance(text, str) or not text.strip():
        return "", ""
    s = text
    # Try to find explicit thought and solution segments
    thought = ""
    solution = ""
    try:
        bt = BEGIN_THINK.search(s)
        et = END_THINK.search(s)
        bs = BEGIN_SOL.search(s)
        es = END_SOL.search(s)
        if bt and et and et.start() > bt.end():
            thought = s[bt.end():et.start()].strip()
        if bs and es and es.start() > bs.end():
            solution = s[bs.end():es.start()].strip()
        # Fallbacks
        if not solution:
            # If no explicit solution markers, try content after end_of_thought
            if et:
                solution = s[et.end():].strip()
            else:
                solution = s.strip()
    except Exception:
        # On any parsing error, keep original as solution
        solution = s.strip()
    return thought, solution


def _as_messages_from_conversations(conversations: Any) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict):
                # Handle keys 'from'/'value' or 'role'/'content'
                role = turn.get('role') or turn.get('from')
                content = turn.get('content') if 'content' in turn else turn.get('value')
                if role in ('user', 'assistant', 'system') and isinstance(content, str):
                    msgs.append({'role': role, 'content': content})
    return msgs


def _insert_augmented_think_into_assistant(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not msgs:
        return msgs
    # Find last user and last assistant
    last_user = ""
    for m in reversed(msgs):
        if m.get('role') == 'user':
            last_user = m.get('content', '') or ''
            break
    last_assistant_idx = None
    for i in range(len(msgs)-1, -1, -1):
        if msgs[i].get('role') == 'assistant':
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        # Create an assistant message if missing
        msgs.append({'role': 'assistant', 'content': ''})
        last_assistant_idx = len(msgs) - 1

    # Extract thought/solution from assistant content
    orig = msgs[last_assistant_idx].get('content', '') or ''
    thought, solution = _extract_thought_and_solution(orig)

    # Build augmented think
    delta = TimeTools.make_delta_tag(TimeTools.random_delta_seconds())
    emotion = EmotionEngine.tag(last_user)
    if not thought:
        # If no thought present, make a minimal one
        thought = "Reflecting on the best way to respond."
    think_wrapped = f"<think>{thought}</think>"
    augmented = f"{delta} {emotion} {think_wrapped}"

    # Replace assistant content with augmented think + original solution
    new_content = augmented
    if solution:
        new_content = f"{augmented}\n\n{solution}"
    msgs[last_assistant_idx]['content'] = _wrap_im(new_content)
    return msgs


def process_openthoughts_and_bespoke(openthoughts_rows: int, bespoke_rows: int) -> Dict[str, pd.DataFrame]:
    datasets_out: Dict[str, pd.DataFrame] = {}
    print(f"[Base] Preparing OpenThoughts ({openthoughts_rows} rows) and Bespoke ({bespoke_rows} rows)...")
    # OpenThoughts
    try:
        ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
        if openthoughts_rows:
            ds = ds.select(range(min(openthoughts_rows, len(ds))))
        print(f"[Base] OpenThoughts loaded: {len(ds)} examples")
        rows = []
        pbar = tqdm(total=len(ds), desc="[Base] OpenThoughts rows", unit="row")
        for ex in ds:
            
            convs = ex.get('conversations')
            msgs = _as_messages_from_conversations(convs)
            if not msgs:
                # Construct from system+conversations raw if possible
                sys_text = ex.get('system', '')
                if sys_text:
                    msgs = [{'role': 'system', 'content': sys_text}]
            msgs = _insert_augmented_think_into_assistant(msgs)
            rows.append({
                'messages': json.dumps(msgs)
            })
            pbar.update(1)
        pbar.close()
        print(f"[Base] OpenThoughts processed into rows: {len(rows)}")
        datasets_out['OpenThoughts-114k'] = pd.DataFrame(rows)
    except Exception as e:
        datasets_out['OpenThoughts-114k'] = pd.DataFrame({'error': [str(e)]})
        print(f"[Base][Error] OpenThoughts: {e}")

    # Bespoke
    try:
        ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
        if bespoke_rows:
            ds = ds.select(range(min(bespoke_rows, len(ds))))
        print(f"[Base] Bespoke loaded: {len(ds)} examples")
        rows = []
        pbar = tqdm(total=len(ds), desc="[Base] Bespoke rows", unit="row")
        for ex in ds:
            convs = ex.get('conversations') or ex.get('messages')
            msgs = _as_messages_from_conversations(convs)
            # If dataset is pairwise text, try to synthesize simple messages
            if not msgs:
                user = ex.get('prompt') or ex.get('instruction') or ''
                assistant = ex.get('completion') or ex.get('response') or ex.get('output') or ''
                if user or assistant:
                    msgs = [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}]
            msgs = _insert_augmented_think_into_assistant(msgs)
            rows.append({
                'messages': json.dumps(msgs)
            })
            pbar.update(1)
        pbar.close()
        print(f"[Base] Bespoke processed into rows: {len(rows)}")
        datasets_out['Bespoke-Stratos-17k'] = pd.DataFrame(rows)
    except Exception as e:
        datasets_out['Bespoke-Stratos-17k'] = pd.DataFrame({'error': [str(e)]})
        print(f"[Base][Error] Bespoke: {e}")

    # Post-process to apply DELTA/think fixes uniformly
    for k in list(datasets_out.keys()):
        print(f"[Base] Post-processing fixes for: {k}")
        datasets_out[k] = process_dataframe(datasets_out[k])

    return datasets_out


def scan_and_fix(input_paths: List[str], output_dir: str, rows_limit: int = 0, include_base: bool = True, base_rows: int = 1000) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    # Optionally include base datasets (OpenThoughts + Bespoke)
    if include_base:
        # Use separate caps for base datasets
        openthoughts_cap = (7680 * 6) if base_rows == 1000 else base_rows or rows_limit
        bespoke_cap = (3840 * 6) if base_rows == 1000 else base_rows or rows_limit
        print(f"[Scan] Including base datasets with caps -> OpenThoughts: {openthoughts_cap}, Bespoke: {bespoke_cap}")
        base = process_openthoughts_and_bespoke(openthoughts_rows=openthoughts_cap, bespoke_rows=bespoke_cap)
        for name, df in base.items():
            out_file = os.path.join(output_dir, f"fixed_{name}.csv")
            try:
                # Apply token-limit filtering before writing
                tokenizer = _load_tokenizer()
                before = len(df)
                df_filtered = filter_by_token_limit(df, tokenizer, limit=8192)
                # Final verification pass
                df_verified, post_dropped = verify_token_limit(df_filtered, tokenizer, limit=8192)
                after = len(df_verified)
                dropped = before - after
                df_filtered.to_csv(out_file, index=False)
                written.append(out_file)
                print(f"[Write] Base dataset written: {out_file} (rows={after}, dropped>{dropped}, post-check>{post_dropped})")
            except Exception as e:
                err_path = os.path.join(output_dir, f"error_{name}.txt")
                with open(err_path, "w", encoding="utf-8") as ef:
                    ef.write(f"Failed to write {name}: {e}\n")
                written.append(err_path)
                print(f"[Write][Error] Base dataset {name}: {e}")

    for path in input_paths:
        if not os.path.exists(path):
            continue
        if os.path.isdir(path):
            # Count CSVs for progress bar
            all_csvs = []
            for root, _, files in os.walk(path):
                for fname in files:
                    if fname.lower().endswith(".csv"):
                        all_csvs.append((root, fname))
            pbar_files = tqdm(total=len(all_csvs), desc="[Scan] Fixed datasets", unit="file")
            for root, fname in all_csvs:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, path)
                out_name = rel.replace("/", "_")
                try:
                    print(f"[Scan] Processing file: {fpath}")
                    df = pd.read_csv(fpath)
                    if rows_limit and len(df) > rows_limit:
                        df = df.head(rows_limit)
                        print(f"[Scan] Row limit applied: {rows_limit} (original larger)")
                    # Always process and filter regardless of row limit branch
                    fixed = process_dataframe(df)
                    tokenizer = _load_tokenizer()
                    before = len(fixed)
                    fixed = filter_by_token_limit(fixed, tokenizer, limit=8192)
                    # Final verification pass
                    fixed, post_dropped = verify_token_limit(fixed, tokenizer, limit=8192)
                    after = len(fixed)
                    dropped = before - after
                    out_file = os.path.join(output_dir, f"fixed_{out_name}")
                    fixed.to_csv(out_file, index=False)
                    written.append(out_file)
                    print(f"[Write] Fixed file written: {out_file} (rows={after}, dropped>{dropped}, post-check>{post_dropped})")
                except Exception as e:
                    err_path = os.path.join(output_dir, f"error_{out_name}.txt")
                    with open(err_path, "w", encoding="utf-8") as ef:
                        ef.write(f"Failed to process {fpath}: {e}\n")
                    written.append(err_path)
                    print(f"[Write][Error] Failed {fpath}: {e}")
                finally:
                    pbar_files.update(1)
            pbar_files.close()
        else:
            if path.lower().endswith(".csv"):
                out_name = os.path.basename(path)
                try:
                    print(f"[Scan] Processing file: {path}")
                    df = pd.read_csv(path)
                    if rows_limit and len(df) > rows_limit:
                        df = df.head(rows_limit)
                        print(f"[Scan] Row limit applied: {rows_limit} (original larger)")
                    fixed = process_dataframe(df)
                    tokenizer = _load_tokenizer()
                    before = len(fixed)
                    fixed = filter_by_token_limit(fixed, tokenizer, limit=8192)
                    fixed, post_dropped = verify_token_limit(fixed, tokenizer, limit=8192)
                    after = len(fixed)
                    dropped = before - after
                    out_file = os.path.join(output_dir, f"fixed_{out_name}")
                    fixed.to_csv(out_file, index=False)
                    written.append(out_file)
                    print(f"[Write] Fixed file written: {out_file} (rows={after}, dropped>{dropped}, post-check>{post_dropped})")
                except Exception as e:
                    err_path = os.path.join(output_dir, f"error_{out_name}.txt")
                    with open(err_path, "w", encoding="utf-8") as ef:
                        ef.write(f"Failed to process {path}: {e}\n")
                    written.append(err_path)
                    print(f"[Write][Error] Failed {path}: {e}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Fix datasets in TrainingData/pools and base sets: DELTA unknowns, think wrapping, emotion tags, and target insertion")
    parser.add_argument("--rows", type=int, default=0, help="Optional limit on rows per file (0 = no limit)")
    parser.add_argument("--out", type=str, default="TrainingData/Fixed", help="Output directory for fixed CSVs")
    parser.add_argument("--paths", type=str, nargs="*", default=[], help="Paths to scan (dirs or files). Defaults to TrainingData/pools")
    parser.add_argument("--no-base", action="store_true", help="Do not include OpenThoughts/Bespoke base datasets")
    parser.add_argument("--base-rows", type=int, default=(7680 * 6), help="Row cap for base datasets (0 = follow --rows). OpenThoughts defaults to 7680*6, Bespoke defaults to 3840*6.")
    args = parser.parse_args()

    default_paths = [
        os.path.join(os.getcwd(), "TrainingData", "pools"),
    ]
    input_paths = args.paths if args.paths else default_paths

    written = scan_and_fix(input_paths, args.out, rows_limit=args.rows, include_base=(not args.no_base), base_rows=args.base_rows)
    print("Wrote:")
    for w in written:
        print(f"- {w}")


if __name__ == "__main__":
    main()
