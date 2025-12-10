from datasets import load_dataset
from ExtraTools import EmotionEngine, TimeTools, prepend_system, inject_identity, get_think_engine
import logging as log
import re
import json
import pandas as pd
import time

class Datsets:
    
    def __init__(self):
        self.system_prompt = (
            "You are Friday, you can think with <think> and do tool calls with <tool>. "
            "Your goal is to be sentient"
        )
        # Initialize the think engine once per program run
        self.engine = get_think_engine()
        
    def return_empathetic(self):
        return self.empathetic
    
    def _augment_user(self, text: str) -> str:
        base = str(text or "")
        emo = EmotionEngine.tag(base)
        delta = TimeTools.make_delta_tag(TimeTools.random_delta_seconds())
        return f"{delta} {emo} {base}".strip()

    def _extract_last_user(self, msgs_json: str) -> str:
        try:
            msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
            if not isinstance(msgs, list):
                return ""
            for m in reversed(msgs):
                if m.get('role') == 'user':
                    return str(m.get('content', '') or '')
        except Exception:
            pass
        return ""

    def _generate_think(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("_generate_think received empty prompt")
        
        
        # Sanitize prompt
        sp = re.sub(r"\[DELTA:\([^)]*\)\]\s*", "", prompt)
        sp = re.sub(r"<[^>]+>\s*", "", sp)
        sp = re.sub(r"\[identity\][^\n]*", "", sp)
        
        thought = self.engine.generate_thought(sp)

        return thought

    def _generate_thinks(self, prompts):
        if not isinstance(prompts, (list, tuple)):
            raise TypeError("prompts must be a list of strings")
        cleaned = []
        for p in prompts:
            sp = re.sub(r"\[DELTA:\([^)]*\)\]\s*", "", p or "")
            sp = re.sub(r"<[^>]+>\s*", "", sp)
            sp = re.sub(r"\[identity\][^\n]*", "", sp)
            cleaned.append(sp)
        thoughts = self.engine.generate_thoughts(cleaned)
        return thoughts

    def _generate_thinks_from_messages(self, msgs_json_list):
        """Generate thoughts using full conversation messages (list of JSON strings or lists).
        Centralizes the strict directive and ensures it's included in the messages."""
        batch_messages = []
        sys_text = (
            "Strict rationale-only mode: Respond only with internal reasoning in first person ('I'). "
            "Your ONLY task is to justify why the given assistant answer fits the user's messages. "
            "Write numbered steps inside <think>…</think>: "
            "1) I restate user intent; 2) I extract key facts/constraints from prior turns; "
            "3) I explain how these lead to the given answer; 4) I note tone/style alignment. "
            "Do not include policies, disclaimers, morality, or safety commentary. Do not quote or paraphrase the final answer."
        )
        for mj in msgs_json_list:
            try:
                msgs = json.loads(mj) if isinstance(mj, str) else mj
            except Exception:
                msgs = []
            if not isinstance(msgs, list):
                msgs = []
            # Ensure a strong system prompt is present (append or replace any system at end)
            has_system = False
            for m in msgs:
                if isinstance(m, dict) and m.get('role') == 'system':
                    m['content'] = sys_text
                    has_system = True
                    break
            if not has_system:
                msgs.append({"role": "system", "content": sys_text})
            batch_messages.append(msgs)
        thoughts = self.engine.generate_thoughts_from_messages(batch_messages)
        return thoughts

    def _augment_df_with_model_think(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or 'messages' not in df.columns:
            return df
        
        out_rows = []
        for _, row in df.iterrows():
            # Check target first (used in your _map_pairwise_conv datasets)
            target = str(row.get('target', '') or '')
            
            # Extract messages
            msgs_raw = row.get('messages', '')
            try:
                msgs = json.loads(msgs_raw) if isinstance(msgs_raw, str) else msgs_raw
            except:
                msgs = []

            # 1. Check if thought already exists in target or last assistant message
            has_think = False
            if '<think>' in target: 
                has_think = True
            
            # Check inside messages for pre-existing think
            for m in msgs:
                if m.get('role') == 'assistant' and '<think>' in str(m.get('content','')):
                    has_think = True
                    break

            if has_think:
                r = row.copy()
                r['think_inserted'] = False
                out_rows.append(r)
                continue

            # 2. Generate Thought
            prompt = self._extract_last_user(msgs)
            if not prompt:
                out_rows.append(row)
                continue
                
            think_block = self._generate_think(prompt)
            
            row = row.copy()
            row['think_inserted'] = True

            # --- FIX 3: Merge thought into content (User -> Assistant w/ CoT) ---
            
            # Case A: Target column exists (common in your pairwise logic)
            if 'target' in row and row['target']:
                # Prepend thought to the target answer
                row['target'] = f"{think_block}\n\n{row['target']}"
            
            # Case B: Assistant response is inside 'messages' (like in HH dataset)
            updated_msgs = False
            if isinstance(msgs, list) and len(msgs) > 0:
                last_msg = msgs[-1]
                if last_msg.get('role') == 'assistant':
                    # Prepend to the existing assistant message
                    original_content = last_msg.get('content', '')
                    last_msg['content'] = f"{think_block}\n\n{original_content}"
                    row['messages'] = json.dumps(msgs)
                    updated_msgs = True

            # Handle case where we have a thought but no assistant target to attach to yet 
            # (If your pipeline expects 'target' to be empty initially, we can set it)
            if not updated_msgs and ('target' not in row or not row['target']):
                 # If no target exists, the thought BECOMES the start of the target
                 row['target'] = think_block

            out_rows.append(row)
            
        return pd.DataFrame(out_rows)
    
    def _map_pairwise_conv(self, ds, user_keys=("instruction","prompt","question","input"), assistant_keys=("output","response","completion","answer"), source_label="generic"):
        t0 = time.time()
        def pick_field(example, keys):
            for k in keys:
                if k in example and isinstance(example[k], str) and example[k].strip():
                    return example[k]
            for k, v in example.items():
                if isinstance(v, str) and v.strip():
                    return v
            return ""
        def conv(example):
            user = pick_field(example, user_keys)
            assistant = pick_field(example, assistant_keys)
            user = inject_identity(user)
            user = self._augment_user(user)
            context = [{"role":"user","content":user}]
            target = assistant
            return {"messages": prepend_system(json.dumps(context), self.system_prompt), "target": target, "source": source_label}
        df = pd.DataFrame(ds.map(conv))
        print(f"{source_label} ready: {len(df)} rows in {int(time.time()-t0)}s")
        return df

    def return_hh(self, num_rows):
        self.hh = load_dataset("Anthropic/hh-rlhf", split="train")
        avail = len(self.hh)
        self.hh = self.hh.select(range(min(num_rows, avail)))
        t0 = time.time()
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
            
            target = ""
            context = []
            
            # If the last message is assistant, that is our target
            if messages and messages[-1]['role'] == 'assistant':
                target = messages.pop()['content']
            
            # Process User messages in context
            for msg in messages:
                if msg['role'] == 'user':
                    msg['content'] = self._augment_user(msg.get('content',''))
                context.append(msg)
                
            return {"messages": prepend_system(json.dumps(context), self.system_prompt), "target": target, "source": "identity"}
        
        hh_df = pd.DataFrame(self.hh.map(hh_conv))
        if len(hh_df) < num_rows:
            hh_df = hh_df.sample(n=num_rows, replace=True, random_state=42)
        print(f"Identity(HH) ready: {len(hh_df)} rows in {int(time.time()-t0)}s")
        return hh_df

    def return_self_cog(self, num_rows):
        ds = load_dataset("modelscope/self-cognition", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(ds, user_keys=("instruction","prompt"), assistant_keys=("output","response"), source_label="self-cognition")
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_openthoughts(self, num_rows):
        ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(ds, user_keys=("prompt","instruction"), assistant_keys=("completion","response","output"), source_label="OpenThoughts-114k")
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_bespoke(self, num_rows):
        ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(ds, user_keys=("prompt","instruction"), assistant_keys=("completion","response","output"), source_label="Bespoke-Stratos-17k")
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_glaive_fc(self, num_rows):
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(ds, user_keys=("prompt","instruction"), assistant_keys=("response","output","completion"), source_label="glaive-function-calling-v2")
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_magicoder(self, num_rows):
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(ds, user_keys=("instruction","prompt"), assistant_keys=("response","output","completion"), source_label="Magicoder-Evol-Instruct-110K")
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_infinity(self, num_rows):
        ds = load_dataset("BAAI/Infinity-Instruct", "7M_core", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(ds, user_keys=("instruction","prompt"), assistant_keys=("response","output","completion"), source_label="Infinity-Instruct-7M_core")
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_open_platypus(self, num_rows):
        ds = load_dataset("garage-bAInd/Open-Platypus", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(ds, user_keys=("instruction","prompt","question"), assistant_keys=("response","output","completion","answer"), source_label="Open-Platypus")
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_empathetic_dialogues(self, num_rows):
        ds = load_dataset("facebook/empathetic_dialogues", split="train", trust_remote_code=True)
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        def conv(example):
            user = example.get("context", "") or example.get("prompt", "")
            assistant = example.get("utterance", "") or example.get("response", "")
            user = inject_identity(user)
            user = self._augment_user(user)
            context = [{"role":"user","content":user}]
            target = assistant
            return {"messages": prepend_system(json.dumps(context), self.system_prompt), "target": target, "source": "empathetic_dialogues"}
        df = pd.DataFrame(ds.map(conv))
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        print(f"empathetic_dialogues ready: {len(df)} rows")
        return df

    def download_all(self, num_rows, add_thoughts: bool = False):
        data = {
            "hh": self.return_hh(num_rows),
            "self_cognition": self.return_self_cog(num_rows),
            "open_thoughts": self.return_openthoughts(num_rows),
            "bespoke_stratos": self.return_bespoke(num_rows),
            "glaive_fc_v2": self.return_glaive_fc(num_rows),
            "magicoder_evol": self.return_magicoder(num_rows),
            "infinity_instruct": self.return_infinity(num_rows),
            "open_platypus": self.return_open_platypus(num_rows),
            "empathetic_dialogues": self.return_empathetic_dialogues(num_rows),
        }
        if add_thoughts:
            for k, v in list(data.items()):
                print(f"--- Generating thoughts for {k} ---")
                data[k] = self._augment_df_with_model_think(v)
        return data