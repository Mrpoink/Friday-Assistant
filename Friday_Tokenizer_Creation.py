from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def create_tokenizer():
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = [
        # Reasoning and tools
        "<think>", "</think>", "<tool_call>", "</tool_call>",
        # Identity marker
        "[identity]",
        # Emotion tags (aligned with EmotionEngine ordered labels)
        "<admiration>", "<amusement>", "<anger>", "<annoyance>", "<approval>", "<caring>",
        "<confusion>", "<curiosity>", "<desire>", "<disappointment>", "<disapproval>",
        "<disgust>", "<embarrassment>", "<excitement>", "<fear>", "<gratitude>", "<grief>",
        "<joy>", "<love>", "<nervousness>", "<optimism>", "<pride>", "<realization>",
        "<relief>", "<remorse>", "<sadness>", "<surprise>", "<neutral>",
        # Time delta categories
        "[DELTA:(SHORT)]", "[DELTA:(MEDIUM)]", "[DELTA:(LONG)]", "[DELTA:(EXTRA LONG)]", "[DELTA:(FOREVER)]",
        # Timestamp and time-of-day markers
        "[TS:", "[TIME:"
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save locally so training can load it
    out_dir = os.path.join("Friday_Tokenizer")
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    return tokenizer

if __name__ == "__main__":
    tok = create_tokenizer()
    print(f"Saved tokenizer with {len(tok)} tokens to ./Friday_Tokenizer")