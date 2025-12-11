import os
from datasets import load_dataset
from ExtraTools import EmotionEngine, TimeTools, prepend_system, inject_identity, get_think_engine
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
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

    def _generate_think(self, prompt: str, target_response: str = "") -> str:
        """
        Calls the engine (Gemini) to generate the thought.
        """
        if not self.engine:
            return ""
        
        # We pass both the prompt (which has tags) and the target response
        # so Gemini knows exactly what logic to hallucinate.
        thought = self.engine.generate(user_text=prompt, target_response=target_response)
        
        # Clean up: Ensure it has tags
        if "<think>" not in thought:
            thought = f"<think>{thought}</think>"
            
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
            """
        You are an expert AI creating a 'Thought Trace' for a smaller student model (Friday, who was developed by Brandon Dean).

INPUT CONTEXT:
The user message will contain metadata tags:
1. [DELTA:...] -> How long the user waited (SHORT, MEDIUM, LONG, FOREVER).
2. [emotion:...] -> The tone the user or model should currently have.
3. TARGET RESPONSE -> The final answer the assistant actually gave.

YOUR TASK:
Generate a <think> block that acts as the internal monologue leading to the TARGET RESPONSE.

RULES FOR THINKING:
- Analyze the [DELTA]. If it is LONG/FOREVER, the thought should reflect worry, guilt, or surprise. If SHORT, it's casual.
- Analyze the [emotion]. The thought must align with this mood.
- DO NOT generate the final response. ONLY generate the content inside the tags.
- Output format: <think> ... reasoning ... </think>

CONTEXTUAL REASONING STRATEGIES (Use the one that fits the input):
1. IF THE INPUT IS LOGICAL OR FACTUAL (OpenThoughts/Enigmata):
   - Break the thought down into clear, atomic, reasoning steps (First Principles).
   - Ask "Why?" AT LEAST THREE TIMES to dig deeper into the causality.
   - When defining terms, use this format: [Concept] --(is_a)--> [Description] --(has_property)--> [Function].

2. IF THE INPUT IS SOCIAL OR CASUAL (SMS):
   - Focus on the *intent* and *relationship* dynamic.
   - Instead of defining terms, reflect on *why* the user feels this way.
   - Keep the internal monologue grounded in personality and memory, not dictionary definitions.

TOOL USE LOGIC:
- If the TARGET RESPONSE implies knowledge not in the conversation (e.g., weather, news, specific facts), assume a tool call was made and successful.
- In the thought, explicitly decide to call the tool to verify this information.
- Format the intent as: <tool_call tool="TOOL_NAME">parameters</tool_call>
- Note: Do not hallucinate the tool's *output* in the thought; simply justify the *need* for the tool that leads to the final answer.

STYLE:
- NEVER apologize or mention being an AI model.
- If the Target Response is concise, explain *why* brevity was chosen.
        """
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

    def _augment_df_with_model_think(self, df: pd.DataFrame, max_workers=25) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or 'messages' not in df.columns:
            return df

        print(f"--- Augmenting {len(df)} rows with max_workers={max_workers} ---")

        # Helper function to process a single row (must be thread-safe)
        def process_row(row_data):
            row = pd.Series(row_data)  # Reconstitute row

            # Check if thought exists in target
            target = str(row.get('target', '') or '')
            if '<think>' in target:
                row['think_inserted'] = False
                return row

            # Check if thought exists in messages
            msgs_raw = row.get('messages', '')
            try:
                msgs = json.loads(msgs_raw) if isinstance(msgs_raw, str) else msgs_raw
            except:
                msgs = []

            for m in msgs:
                if m.get('role') == 'assistant' and '<think>' in str(m.get('content','')):
                    row['think_inserted'] = False
                    return row

            # Generate Thought
            prompt = self._extract_last_user(msgs)
            if not prompt:
                return row

            # CALL GEMINI (This is the slow part that we parallelize)
            think_block = self._generate_think(prompt)

            row['think_inserted'] = True

            # Merge Logic
            if 'target' in row and row['target']:
                row['target'] = f"{think_block}\n\n{row['target']}"

            updated_msgs = False
            if isinstance(msgs, list) and len(msgs) > 0:
                last_msg = msgs[-1]
                if last_msg.get('role') == 'assistant':
                    original_content = last_msg.get('content', '')
                    last_msg['content'] = f"{think_block}\n\n{original_content}"
                    row['messages'] = json.dumps(msgs)
                    updated_msgs = True

            if not updated_msgs and ('target' not in row or not row['target']):
                 row['target'] = think_block

            return row

        # PARALLEL EXECUTION
        new_rows = []
        rows_to_process = df.to_dict('records')

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(process_row, r): r for r in rows_to_process}

            completed_count = 0
            total = len(rows_to_process)

            for future in concurrent.futures.as_completed(future_to_row):
                try:
                    result_row = future.result()
                    new_rows.append(result_row)

                    completed_count += 1
                    if completed_count % 100 == 0:
                        print(f"Processed {completed_count}/{total} rows...", end='\r')

                except Exception as e:
                    print(f"Row failed: {e}")
                    new_rows.append(future_to_row[future])

        print(f"\nFinished processing {len(new_rows)} rows.")
        return pd.DataFrame(new_rows)
    
    def _map_pairwise_conv(self, ds, user_keys=("instruction","prompt","question","input"), assistant_keys=("output","response","completion","answer"), source_label="generic"):
        t0 = time.time()
        def pick_field(example, keys):
            for k in keys:
                if k in example:
                    val = example[k]
                    # Robustness check: Ensure we return a string
                    if isinstance(val, str) and val.strip():
                        return val
                    # If it's a list (like messages), try to extract user/assistant text blindly?
                    # For now, just preventing the crash is the priority.
            return ""
        def conv(example):
            user = pick_field(example, user_keys)
            assistant = pick_field(example, assistant_keys)
            user = inject_identity(user)
            user = self._augment_user(user)
            context = [{"role":"user","content":user}]
            target = assistant
            # Add a lightweight hint to the system so the student understands the context
            hint = "[dataset:" + ("tools" if "glaive" in source_label else "code") + "]"
            sys_text = f"{self.system_prompt}\n{hint}"
            return {"messages": prepend_system(json.dumps(context), sys_text), "target": target, "source": source_label}
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
        # ERROR FIX: This dataset uses a 'messages' list, not flat columns.
        ds = load_dataset("modelscope/self-cognition", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        
        def conv(example):
            # Extract from 'messages' list
            msgs = example.get("messages", [])
            target = ""
            user_content = ""
            
            # Simple extraction: Last assistant is target, last user is context
            if isinstance(msgs, list):
                for m in reversed(msgs):
                    if m.get('role') == 'assistant' and not target:
                        target = m.get('content', '')
                    if m.get('role') == 'user' and not user_content:
                        user_content = m.get('content', '')
            
            user_content = inject_identity(user_content)
            user_content = self._augment_user(user_content)
            
            # Rebuild standardized messages
            context = [{"role":"user", "content": user_content}]
            return {"messages": prepend_system(json.dumps(context), self.system_prompt), "target": target, "source": "self-cognition"}

        df = pd.DataFrame(ds.map(conv))
        if len(df) < num_rows and len(df) > 0:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_openthoughts(self, num_rows):
        ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        # FIX: Added 'question' to user_keys
        df = self._map_pairwise_conv(
            ds, 
            user_keys=("system", "question", "prompt", "instruction"), 
            assistant_keys=("response", "completion", "output"), 
            source_label="OpenThoughts-114k"
        )
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_bespoke(self, num_rows):
        ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        # FIX: Added 'conversations' checking logic implicitly via robust keys
        # If this dataset uses 'conversations' (list), _map_pairwise_conv might still fail unless updated.
        # Assuming it uses standard keys for now, but adding 'messages' handling is safer.
        # Ideally, check if it has 'conversations' column first. 
        # For now, adding 'content' and 'query' just in case.

        def conv(example):
            # Check for 'messages' OR 'conversations'
            msgs = example.get("messages", example.get("conversations", []))
            
            target = ""
            user_content = ""
            
            # Extract content from list of dicts
            if isinstance(msgs, list):
                for m in reversed(msgs):
                    role = m.get('role', '')
                    content = m.get('content', '') or m.get('value', '') # sometimes key is 'value'
                    
                    if role == 'assistant' and not target:
                        target = content
                    if role == 'user' and not user_content:
                        user_content = content
            
            # Fallback for flat columns if the list method failed (just in case)
            if not user_content:
                 user_content = example.get('prompt', example.get('instruction', example.get('query', '')))
            if not target:
                 target = example.get('completion', example.get('response', example.get('output', '')))

            user_content = inject_identity(user_content)
            user_content = self._augment_user(user_content)
            
            # Rebuild standardized messages
            context = [{"role":"user", "content": user_content}]
            return {"messages": prepend_system(json.dumps(context), self.system_prompt), "target": target, "source": "Bespoke-Stratos-17k"}
        df = pd.DataFrame(ds.map(conv))
        if len(df) < num_rows and len(df) > 0:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_glaive_fc(self, num_rows):
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        # FIX: Glaive uses 'chat' (user) and 'system' (assistant)
        df = self._map_pairwise_conv(
            ds, 
            user_keys=("chat", "prompt", "instruction"), 
            assistant_keys=("system", "response", "output", "completion"), 
            source_label="glaive-function-calling-v2"
        )
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_magicoder(self, num_rows):
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(
            ds,
            user_keys=("instruction","prompt"),
            assistant_keys=("response","output","completion"),
            source_label="Magicoder-Evol-Instruct-110K"
        )
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_infinity(self, num_rows):
        ds = load_dataset("BAAI/Infinity-Instruct", "7M_core", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(
            ds,
            user_keys=("instruction","prompt"),
            assistant_keys=("response","output","completion"),
            source_label="Infinity-Instruct-7M_core"
        )
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_open_platypus(self, num_rows):
        ds = load_dataset("garage-bAInd/Open-Platypus", split="train")
        avail = len(ds)
        ds = ds.select(range(min(num_rows, avail)))
        df = self._map_pairwise_conv(
            ds,
            user_keys=("instruction","prompt","question"),
            assistant_keys=("response","output","completion","answer"),
            source_label="Open-Platypus"
        )
        if len(df) < num_rows:
            df = df.sample(n=num_rows, replace=True, random_state=42)
        return df

    def return_sms(self, num_rows):
        """Load SMS-style training data from TrainingData/training_data.csv and format for RLAIF.
        Expects columns: messages (JSON string) and target (assistant visible reply).
        - Ensures user messages include [DELTA:(..)] and <emotion>
        - Prepends a system prompt
        - Leaves target as-is (visible reply); thought insertion happens later upstream
        """
        tpath = os.path.join(os.getcwd(), "TrainingData", "training_data.csv")
        if not os.path.isfile(tpath):
            print(f"SMS training_data.csv not found: {tpath}")
            return pd.DataFrame(columns=["messages","target","source"]) 

        try:
            df = pd.read_csv(tpath)
        except Exception as e:
            print(f"Failed to read training_data.csv: {e}")
            return pd.DataFrame(columns=["messages","target","source"]) 

        rows = []
        take = min(num_rows, len(df))
        df = df.head(take)
        for _, r in df.iterrows():
            msgs_raw = r.get("messages", "[]")
            target = str(r.get("target", "") or "")
            # Parse messages JSON
            try:
                msgs = json.loads(msgs_raw) if isinstance(msgs_raw, str) else msgs_raw
            except Exception:
                msgs = []
            if not isinstance(msgs, list):
                msgs = []
            # Ensure user turns carry delta/emotion
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    m["content"] = self._augment_user(m.get("content", ""))
            # Prepend/refresh system prompt
            msgs_json = prepend_system(json.dumps(msgs), self.system_prompt)
            rows.append({
                "messages": msgs_json,
                "target": target,
                "source": "sms"
            })
        df_out = pd.DataFrame(rows)
        # If fewer than requested, upsample
        if len(df_out) < num_rows and len(df_out) > 0:
            df_out = df_out.sample(n=num_rows, replace=True, random_state=42)
        print(f"sms ready (training_data.csv): {len(df_out)} rows")
        return df_out

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
            "sms": self.return_sms(num_rows),
        }
        if add_thoughts:
            for k, v in list(data.items()):
                print(f"--- Generating thoughts for {k} ---")
                data[k] = self._augment_df_with_model_think(v, max_workers=25)
        return data