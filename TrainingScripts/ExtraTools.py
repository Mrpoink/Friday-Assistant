import os
os.environ["GEMINI_API_KEY"] = "AIzaSyCO1iuF6k9sJ7egnSoBG6RSqGKnRlTA-_E"  # Replace with your actual API key

import random
import re
from datetime import timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
from transformers import pipeline
from google import genai
from google.genai import types
import os
import time

GEMINI_API_KEY = "GEMINI_API_KEY"  # Replace with your actual API key

# === NEW: Gemini Teacher Class ===
class GeminiThinker:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.api_key = GEMINI_API_KEY
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY not found in environment variables.")
            self.client = None
        else:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        self.model_name = model_name

        # This is the "Teacher" Prompt that explains your custom tags to Gemini
        self.system_instruction = """
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

    def generate(self, user_text, target_response):
        if not self.client:
            return "<think>API Key missing. Could not generate thought.</think>"

        # We feed Gemini the User's input AND the target answer so it knows what to justify
        prompt = f"""
        User Input: {user_text}
        
        Target Response to justify: {target_response}
        
        Generate the <think> block that leads to this response:
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.7,
                    max_output_tokens=2048
                )
            )
            # Guard against None text; coerce to string and ensure <think> wrapper
            text = getattr(response, "text", "")
            safe = str(text or "").strip()
            if not safe:
                return "<think>I assume the answer, however I cannot be entirely sure</think>"
            if "<think>" not in safe:
                safe = f"<think>{safe}</think>"
            return safe
            
        except Exception as e:
            print(f"Gemini Error: {e}")
            time.sleep(2) # Brief pause on error (rate limit backoff)
            return "<think>Generation failed.</think>"

    # Compatibility methods mirroring ThinkEngine API
    def generate_thought(self, prompt: str, max_new_tokens: int = 512) -> str:
        # Single-input path: treat entire prompt as user text, no target
        return self.generate(user_text=str(prompt or ""), target_response="")

    def generate_thoughts(self, prompts, max_new_tokens: int = 512):
        if not isinstance(prompts, (list, tuple)):
            raise TypeError("prompts must be a list of strings")
        results = []
        for p in prompts:
            results.append(self.generate_thought(str(p or ""), max_new_tokens=max_new_tokens))
        return results

    def generate_thought_from_messages(self, messages, max_new_tokens: int = 512) -> str:
        # Flatten chat messages into a single prompt including roles
        try:
            msgs = messages if isinstance(messages, list) else []
        except Exception:
            msgs = []
        parts = []
        target_reply = ""
        for m in msgs:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            parts.append(f"{role.upper()}:\n{content}\n")
            if role == "assistant":
                target_reply = content  # last assistant content becomes target
        user_text = "\n".join(parts)
        return self.generate(user_text=user_text, target_response=target_reply)

    def generate_thoughts_from_messages(self, batch_messages, max_new_tokens: int = 512):
        if not isinstance(batch_messages, (list, tuple)):
            raise TypeError("batch_messages must be a list of message lists")
        results = []
        for msgs in batch_messages:
            results.append(self.generate_thought_from_messages(msgs, max_new_tokens=max_new_tokens))
        return results

class TimeTools:
    CATEGORIES = [
        ("SHORT", 60*60),           # < 1 hour
        ("MEDIUM", 5*60*60),       # < 5 hours
        ("LONG", 12*60*60),        # < 12 hours
        ("EXTRA LONG", 24*60*60),  # < 24 hours
        ("FOREVER", 48*60*60),     # > 48 hours
    ]

    @staticmethod
    def categorize(seconds: int) -> str:
        if seconds < TimeTools.CATEGORIES[0][1]:
            return "SHORT"
        if seconds < TimeTools.CATEGORIES[1][1]:
            return "MEDIUM"
        if seconds < TimeTools.CATEGORIES[2][1]:
            return "LONG"
        if seconds < TimeTools.CATEGORIES[3][1]:
            return "EXTRA LONG"
        return "FOREVER"

    @staticmethod
    def random_delta_seconds() -> int:
        buckets = [
            (0, 60*60),
            (60*60, 5*60*60),
            (5*60*60, 12*60*60),
            (12*60*60, 24*60*60),
            (24*60*60, 72*60*60)
        ]
        low, high = random.choice(buckets)
        return random.randint(low, high)

    @staticmethod
    def make_delta_tag(seconds: int) -> str:
        cat = TimeTools.categorize(seconds)
        return f"[DELTA:({cat})]"

    @staticmethod
    def parse_timestamp(text: str):
        from datetime import datetime
        if not text:
            return None
        candidates = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%m/%d/%Y %I:%M %p",
            "%m/%d/%Y %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ]
        s = text.strip()
        for fmt in candidates:
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    @staticmethod
    def delta_seconds(prev_ts, next_ts) -> int:
        if prev_ts and next_ts:
            try:
                return max(0, int((next_ts - prev_ts).total_seconds()))
            except Exception:
                return TimeTools.random_delta_seconds()
        return TimeTools.random_delta_seconds()

_EMOTION_PIPELINE = None

def _get_emotion_pipeline():
    global _EMOTION_PIPELINE
    if _EMOTION_PIPELINE is None:
        try:
            use_device = 0 if torch.cuda.is_available() else -1
            _EMOTION_PIPELINE = pipeline(
                task="text-classification",
                model="SamLowe/roberta-base-go_emotions",
                top_k=None,
                device=use_device
            )
        except Exception:
            _EMOTION_PIPELINE = None
    return _EMOTION_PIPELINE

class EmotionEngine:
    ordered_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    @staticmethod
    def get_emotion_data(text):
        if not text or not isinstance(text, str):
            return "neutral", np.zeros(len(EmotionEngine.ordered_labels))
        pl = _get_emotion_pipeline()
        if pl is None:
            return "neutral", np.zeros(len(EmotionEngine.ordered_labels))
        try:
            output = pl(text, truncation=True, max_length=512)
        except Exception:
            safe_text = text[:4000]
            output = pl(safe_text)
        scores_dict = {item['label']: item['score'] for item in output[0]}
        vector = np.array([scores_dict.get(label, 0.0) for label in EmotionEngine.ordered_labels])
        dominant_emotion = max(scores_dict, key=scores_dict.get)
        return dominant_emotion, vector

    @staticmethod
    def tag(text: str) -> str:
        label, _ = EmotionEngine.get_emotion_data(text)
        return f"<{label}>"
    
class ThinkEngine:
    def __init__(self, model_name: str = "phi3.5:3.8b"):
        pass
        # # Use Ollama as the backing inference engine.
        # # Ensure the model is available in Ollama via `ollama pull`.
        # import os
        # import ollama
        # self._ollama = ollama
        # # Allow overriding model and VRAM-affecting options via env
        # self.model_name = os.getenv("FRIDAY_OLLAMA_MODEL", model_name)
        # self.num_ctx = int(os.getenv("FRIDAY_OLLAMA_NUM_CTX", "1024"))
        # self.default_num_predict = int(os.getenv("FRIDAY_OLLAMA_NUM_PREDICT", "512"))
        # self.temperature = float(os.getenv("FRIDAY_OLLAMA_TEMPERATURE", "0.7"))
        # # keep_alive=0 forces unload after each call to free VRAM
        # self.keep_alive = os.getenv("FRIDAY_OLLAMA_KEEP_ALIVE", "0")

    def _chat(self, prompt: str, max_new_tokens: int = 512) -> str:
        system_text = (
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
   - Ask "Why?" at least three times to dig deeper into the causality.
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
        try:
            # Respect env-tuned limits for lower VRAM usage
            predict = min(int(max_new_tokens or 0) or self.default_num_predict, self.default_num_predict)
            data = self._ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt or ""},
                ],
                options={
                    "num_predict": predict,
                    "num_ctx": self.num_ctx,
                    "temperature": self.temperature,
                },
                keep_alive=self.keep_alive,
            )
            content = data.get("message", {}).get("content", "")
            return str(content or "").strip()
        except Exception as e:
            # Return a minimal think block on failure
            return f"<think>Internal reasoning unavailable: {e}</think>"

    def generate_thought(self, prompt: str, max_new_tokens: int = 512) -> str:
        return self._chat(prompt, max_new_tokens=max_new_tokens)

    def generate_thoughts(self, prompts, max_new_tokens: int = 512):
        if not isinstance(prompts, (list, tuple)):
            raise TypeError("prompts must be a list of strings")
        results = []
        for p in prompts:
            results.append(self._chat(p or "", max_new_tokens=max_new_tokens))
        return results

    def generate_thought_from_messages(self, messages, max_new_tokens: int = 512) -> str:
        """Generate a thought using full chat history messages (list of dicts)."""
        try:
            predict = min(int(max_new_tokens or 0) or self.default_num_predict, self.default_num_predict)
            data = self._ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "num_predict": predict,
                    "num_ctx": self.num_ctx,
                    "temperature": self.temperature,
                },
                keep_alive=self.keep_alive,
            )
            return str(data.get("message", {}).get("content", "") or "").strip()
        except Exception as e:
            return f"<think>Internal reasoning unavailable: {e}</think>"

    def generate_thoughts_from_messages(self, batch_messages, max_new_tokens: int = 512):
        if not isinstance(batch_messages, (list, tuple)):
            raise TypeError("batch_messages must be a list of message lists")
        results = []
        for msgs in batch_messages:
            results.append(self.generate_thought_from_messages(msgs, max_new_tokens=max_new_tokens))
        return results

_THINK_ENGINE = None

def get_think_engine(model_name: str = "phi3.5:3.8b") -> ThinkEngine:
    if GEMINI_API_KEY:
        print("--- Loaded Gemini 1.5 Flash Teacher ---")
        return GeminiThinker()
    else:
        print("--- No API Key found, using dummy/local engine ---")
        global _THINK_ENGINE
        if _THINK_ENGINE is None or getattr(_THINK_ENGINE, 'model_name', None) != model_name:
            _THINK_ENGINE = ThinkEngine(model_name=model_name)
        return _THINK_ENGINE
    
class get_time:
    def __init__(self, real_time = None):
        if real_time:
            self.time = self.parse_time(real_time)
        else:
            self.time = self.make_time()
            
    def parse_time(self, real_time):
        hour, minute = real_time.split(":")
        return f"{int(hour)}: {int(minute)}"
        
    def make_time(self, previous_time = None):
        if previous_time:
            hour, minute = previous_time.split(":")
            minute = int(minute) + random.randint(5, 20)
            if minute >= 60:
                minute = minute - 60
                hour = int(hour) + 1
            hour = int(hour) + random.randint(0, 48)
            if hour >= 24:
                hour = hour - 24
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        return f"{hour:02}:{minute:02}"
    
    def return_time(self):
        hour, minute = self.time.split(":")
        hour = int(hour)
        if hour >= 0 and hour < 6:
            return "Late night"
        elif hour >= 6 and hour < 12:
            return "Morning"
        elif hour >= 12 and hour < 15:
            return "Afternoon"
        elif hour >= 15 and hour < 18:
            return "Late day"
        elif hour >= 18 and hour < 21:
            return "Early evening"
        else:
            return "Late evening"
        
def prepend_system(msgs_json: str, SYSTEM_PROMPT) -> str:
    try:
        msgs = json.loads(msgs_json) if isinstance(msgs_json, str) else msgs_json
    except Exception:
        msgs = []
    if not isinstance(msgs, list):
        msgs = []
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
    
def inject_identity(text: str) -> str:
    s = str(text)
    s = re.sub(r"\bI am an AI\b|\bI'm an AI\b|\bI am a(?:n)? (?:AI|large language model)\b", "I am Friday", s, flags=re.IGNORECASE)
    s = re.sub(r"\bAs an AI\b|\bAs a(?:n)? (?:AI|language model)\b", "As Friday", s, flags=re.IGNORECASE)
    if "server:self local machine" not in s.lower():
        s += "\n[identity] server:self local machine"
    return s