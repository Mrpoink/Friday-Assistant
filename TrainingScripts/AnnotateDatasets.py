import os
import json
import pandas as pd
from typing import Optional

from ExtraTools import TimeTools, EmotionEngine, prepend_system


def _ensure_delta_and_emotion(user_text: str) -> str:
    """Add [DELTA:(..)] and <emotion> tags to a user message if missing."""
    s = str(user_text or "")
    has_delta = "[DELTA:" in s
    has_emotion = "<" in s and ">" in s  # lightweight check for existing emotion tags
    delta_tag = TimeTools.make_delta_tag(TimeTools.random_delta_seconds()) if not has_delta else ""
    emo_tag = EmotionEngine.tag(s) if not has_emotion else ""
    s = f"{delta_tag} {emo_tag} {s}".strip()
    return s


def _extract_last_assistant(messages: list) -> str:
    """Return the content of the last assistant message if present."""
    if not isinstance(messages, list):
        return ""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content", "") or "")
    return ""


def annotate_csv(input_path: str, output_path: Optional[str] = None) -> str:
    """Annotate a dataset CSV by:
    - Ensuring user messages include [DELTA:(..)] and <emotion>
    - If `target` has only a <think> block, append the visible assistant reply from messages when available
    - If neither target nor messages have a visible assistant reply, keep the <think> as target

    Returns the path to the written CSV.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    df = pd.read_csv(input_path)
    if len(df) == 0:
        out = output_path or input_path
        df.to_csv(out, index=False, encoding="utf-8")
        return out

    rows = []
    for _, row in df.iterrows():
        r = row.copy()
        # Normalize messages: add delta/emotion to user turns
        msgs_raw = r.get("messages", "[]")
        try:
            msgs = json.loads(msgs_raw) if isinstance(msgs_raw, str) else msgs_raw
        except Exception:
            msgs = []
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    m["content"] = _ensure_delta_and_emotion(m.get("content", ""))
            # Ensure a system prompt exists at the beginning if absent
            has_system = any(isinstance(m, dict) and m.get("role") == "system" for m in msgs)
            if not has_system:
                msgs = json.loads(prepend_system(json.dumps(msgs), "You are Friday. Follow tags and provide helpful answers."))
            r["messages"] = json.dumps(msgs)

        # Repair target: if it contains only a <think>, append assistant visible reply when present
        target = str(r.get("target", "") or "")
        last_assistant = _extract_last_assistant(msgs if isinstance(msgs, list) else [])
        if "<think>" in target and last_assistant.strip():
            # Append the assistant reply after the think block
            r["target"] = f"{target.strip()}\n\n{last_assistant.strip()}"
        elif not target.strip() and last_assistant.strip():
            # No target yet; use assistant reply directly
            r["target"] = last_assistant.strip()
        # else: keep existing target (may be <think> only)

        rows.append(r)

    out_path = output_path or input_path
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Annotate dataset CSVs with deltas/emotions and visible replies")
    p.add_argument("input", help="Path to input CSV (e.g., TrainingData/pools/sms.csv)")
    p.add_argument("--output", help="Optional path for output CSV; defaults to overwrite input")
    args = p.parse_args()
    out = annotate_csv(args.input, args.output)
    print(f"Annotated CSV written: {out}")