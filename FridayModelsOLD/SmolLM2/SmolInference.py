from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import sys
import importlib.util
import os

# Block bitsandbytes to avoid torch.compile error on Python 3.14+
original_find_spec = importlib.util.find_spec
def patched_find_spec(name, package=None):
    if name == "bitsandbytes" or name.startswith("bitsandbytes."):
        return None
    return original_find_spec(name, package)

importlib.util.find_spec = patched_find_spec


checkpoint_path = "prompt_classifier_Qwen/checkpoint-440"

# Read base model id used during LoRA training from adapter config
adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
with open(adapter_config_path, "r") as f:
    adapter_config = json.load(f)
base_model_id = adapter_config.get("base_model_name_or_path")

print(f"Loading base model '{base_model_id}' with adapter from {checkpoint_path}...")

# Load tokenizer from the adapter directory to include any added tokens
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Load the base model, then apply the LoRA adapter
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,  # Use fp16 for faster inference
    device_map="cuda"
)

# Align embeddings with tokenizer length if needed
if hasattr(model, "resize_token_embeddings"):
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

# Attach LoRA adapter
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()

print("Model + adapter loaded successfully!\n")

def chat(user_message, conversation_history=None):
    
    if conversation_history is None:
        conversation_history = []
    
    conversation_history.append({"role": "user", "content": user_message})
    
    text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True # Adds <|im_start|>assistant to prompt the response
    )
    
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1028,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    response = response.strip()
    
    
    conversation_history.append({"role": "assistant", "content": response})
    
    return response, conversation_history


def interactive_chat():
    
    
    conversation_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_history = []
            print("\nConversation history cleared!\n")
            continue
        
        if not user_input:
            continue
        
        print("Friday: ", end="", flush=True)
        response, conversation_history = chat(user_input, conversation_history)
        print(response + "\n")


if __name__ == "__main__":
    
    interactive_chat()
    