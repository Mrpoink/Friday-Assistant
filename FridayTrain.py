from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback, BitsAndBytesConfig
from datasets import Dataset, load_dataset, concatenate_datasets
import torch
import torch.nn.functional as F
import json
import os
import sys
import platform
import evaluate
import numpy as np
import pandas as pd
import gc
import re

import sys
import importlib.util

from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def weigh_datasets(style, logic, memory, tools, self_aware, identity, logic_text,
                   style_weight, logic_weight, memory_weight, tools_weight, self_aware_weight, identity_weight, logic_text_weight):
    def get_weighted_subset(df, weight):
        if len(df) == 0:
            return df
            
        # 1. Calculate exactly how many rows we want
        target_size = int(len(df) * weight)
        
        # 2. Use .sample() to handle both upsampling and downsampling
        # replace=True allows it to pick the same row twice (necessary if weight > 1.0)
        # replace=False is safer if weight < 1.0 (prevents duplicates in downsampling), 
        # but replace=True works for both if you don't mind duplicates. 
        
        # Logic: If we want MORE rows than exist, we MUST use replace=True.
        # If we want FEWER, we generally prefer replace=False (unique rows).
        do_replace = target_size > len(df)
        
        return df.sample(n=target_size, replace=do_replace, random_state=42)

    return (
        get_weighted_subset(style, style_weight),
        get_weighted_subset(logic, logic_weight),
        get_weighted_subset(memory, memory_weight),
        get_weighted_subset(tools, tools_weight),
        get_weighted_subset(self_aware, self_aware_weight),
        get_weighted_subset(identity, identity_weight),
        get_weighted_subset(logic_text, logic_text_weight)
    )
    
    
    
print("Loading epoch datasets...")
import argparse
parser = argparse.ArgumentParser(description="Friday training with epoch CSVs")
parser.add_argument("--epoch_dir", type=str, default="TrainingData/epochs", help="Directory containing epoch_*.csv files")
parser.add_argument("--epochs", type=int, default=0, help="Number of epochs to train over (files epoch_1..N.csv). If 0, auto-detect from epoch_dir")
parser.add_argument("--prefer_think_tool", action="store_true", help="Upsample examples containing <think> or <tool_call>")
args, _ = parser.parse_known_args()

def load_epoch_df(epoch_idx: int):
    path = os.path.join(args.epoch_dir, f"epoch_{epoch_idx}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Epoch file not found: {path}")
    df = pd.read_csv(path)
    if 'source' not in df.columns:
        df['source'] = 'friday'
    return df

print("Preparing initial epoch dataset...")

# Prefer locally saved tokenizer with special tokens; fallback to base
tok_dir = os.path.join("Friday_Tokenizer")
if os.path.exists(tok_dir):
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    # Minimal specials if local tokenizer not found
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<think>", "</think>", "<tool_call>", "<\/tool_call>", "[identity]"
        ]
    })
tokenizer.model_max_length = 8192
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Combine all datasets
def prefer_examples(df, enabled: bool):
    if not enabled:
        return df
    mask = df['target'].astype(str).str.contains('<think>|<tool_call>', regex=True) | df['messages'].astype(str).str.contains('<think>|<tool_call>', regex=True)
    preferred = df[mask]
    non_preferred = df[~mask]
    if len(preferred) > 0:
        upsampled = preferred.sample(n=min(len(df), len(preferred)*2), replace=True, random_state=42)
        combined = pd.concat([upsampled, non_preferred], ignore_index=True)
        return combined.sample(frac=1.0, random_state=42)
    return df

auto_epochs = args.epochs
if auto_epochs <= 0:
    # auto-detect highest epoch_N.csv in epoch_dir
    try:
        files = [f for f in os.listdir(args.epoch_dir) if re.match(r"epoch_(\d+)\.csv", f)]
        nums = [int(re.match(r"epoch_(\d+)\.csv", f).group(1)) for f in files]
        auto_epochs = max(nums) if nums else 1
        print(f"Auto-detected epochs: {auto_epochs}")
    except Exception:
        auto_epochs = 1

current_epoch_df = prefer_examples(load_epoch_df(1), args.prefer_think_tool)
train_dataset = Dataset.from_pandas(current_epoch_df).shuffle(seed=42)
test_dataset = train_dataset.train_test_split(test_size=0.02)['test']


def format_conversation(example):
    # 1. Parse the JSON string from your CSV
    messages = json.loads(example['messages'])
    source = example['source']
    
    # 2. Pick the correct Persona (System Prompt) — always mention Friday and Brandon
    if source == 'logic':
        sys_prompt = "You are Friday, a helpful coding assistant created by Brandon Dean."
    elif source == 'tools':
        sys_prompt = "You are Friday, created by Brandon Dean. You have access to tools."
    elif source == 'logic_text':
        sys_prompt = "You are Friday, created by Brandon Dean, a logical and critical thinker."
    elif source == 'rag_skill':
        sys_prompt = "You are Friday, created by Brandon Dean. Use the provided context to think step-by-step and answer precisely."
    elif source in ('self_aware','identity'):
        sys_prompt = "You are Friday, a helpful AI assistant created by Brandon Dean."
    else:
        sys_prompt = "You are Friday, an AI assistant created by Brandon Dean."

    # 3. Build the clean message list
    # Start with the System Prompt
    conversation_payload = [{"role": "system", "content": sys_prompt}]
    
    # Add the user/assistant history (filtering out old system prompts)
    for msg in messages:
        if msg.get('role') != "system":
            role = msg.get('role', 'user')
            content = msg.get('content')
            if content is None:
                content = ""
            else:
                content = str(content)
            conversation_payload.append({"role": role, "content": content})
    
    # 4. Add the Target Response (This is what the model trains on)
    assistant_target = example.get('target', '')
    if assistant_target is None:
        assistant_target = ""
    else:
        assistant_target = str(assistant_target)
    conversation_payload.append({"role": "assistant", "content": assistant_target})
    
    # 5. Apply the template
    # tokenize=False gives you the raw string with all special tokens (<|im_start|>) inserted correctly
    text = tokenizer.apply_chat_template(conversation_payload, tokenize=False)
    # Track assistant span for masking: assistant text appears as the last turn content
    # We locate the exact assistant_target substring occurrence at the end.
    try:
        start_idx = text.rfind(assistant_target)
        end_idx = start_idx + len(assistant_target) if start_idx != -1 else -1
    except Exception:
        start_idx, end_idx = -1, -1
    return {"text": text, "assistant_start": start_idx, "assistant_end": end_idx}

formatted_train = train_dataset.map(format_conversation)
formatted_test = test_dataset.map(format_conversation)

metric = evaluate.load('accuracy')

def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def find_lr(epoch_index):
    new_lr = 5e-5 * (0.5 ** (epoch_index - 1)) # Epoch 1: 5e-5, Epoch 2: 2.5e-5, Epoch 3: 1.25e-5
    return new_lr

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["embed_tokens", "lm_head"]
)


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    attn_implementation="sdpa"
)

model.enable_input_require_grads()
# Resize embeddings to accommodate added special tokens
model.resize_token_embeddings(len(tokenizer))

print("Applying LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.config.use_cache = False

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=8192,
        return_attention_mask=True,
        return_offsets_mapping=True
    )
    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]
    labels = []
    starts = examples.get("assistant_start", [-1]*len(input_ids))
    ends = examples.get("assistant_end", [-1]*len(input_ids))
    for i in range(len(input_ids)):
        seq_labels = input_ids[i].copy()
        s = starts[i]
        e = ends[i]
        if s is None:
            s = -1
        if e is None:
            e = -1
        for j, (a, b) in enumerate(offsets[i]):
            if s == -1 or e == -1:
                # Mask all if assistant span not found
                seq_labels[j] = -100
            else:
                # Mask tokens whose character span lies outside assistant span
                if not (a >= s and b <= e):
                    seq_labels[j] = -100
        labels.append(seq_labels)
    tokenized.pop("offset_mapping", None)
    tokenized["labels"] = labels
    return tokenized


tokenized_train = formatted_train.map(tokenize_function, batched=True, remove_columns=formatted_train.column_names)
tokenized_test = formatted_test.map(tokenize_function, batched=True, remove_columns=formatted_test.column_names)


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8
)


is_windows = platform.system().lower() == "windows"


training_args = TrainingArguments(
    output_dir="Friday1.3-Coder-1.5B-LoRA-Instruct-Thinking-SA",
    eval_strategy='steps',
    eval_steps=100,
    save_strategy='steps',
    save_steps=500,
    load_best_model_at_end=True,
    push_to_hub=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    report_to=[],
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    fp16=False,
    bf16=True,
    max_grad_norm=0.35,
    logging_steps=50,
    save_total_limit=3,
    dataloader_pin_memory=True,
    dataloader_num_workers=0 if is_windows else 2,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    weight_decay=0.01,
    greater_is_better=False,
    group_by_length=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
)

# Loop over epochs, reloading per-epoch dataset each time
for epoch_idx in range(1, auto_epochs + 1):
    print(f"Preparing epoch {epoch_idx} dataset...")
    df_epoch = prefer_examples(load_epoch_df(epoch_idx), args.prefer_think_tool)
    ds_epoch = Dataset.from_pandas(df_epoch).shuffle(seed=42)
    split = ds_epoch.train_test_split(test_size=0.02)
    formatted_train = split['train'].map(format_conversation)
    formatted_test = split['test'].map(format_conversation)
    tokenized_train = formatted_train.map(tokenize_function, batched=True, remove_columns=formatted_train.column_names)
    tokenized_test = formatted_test.map(tokenize_function, batched=True, remove_columns=formatted_test.column_names)
    trainer.train_dataset = tokenized_train
    trainer.eval_dataset = tokenized_test
    current_lr = find_lr(epoch_idx)
    trainer.args.learning_rate = current_lr
    print(f"Setting learning rate to {current_lr}")
    print(f"Starting training for epoch {epoch_idx}...")
    trainer.train()
    save_path = f"./Friday-Epoch-{epoch_idx}"
    trainer.save_model(save_path)
    print(f"Saved epoch model to {save_path}")
    del df_epoch, ds_epoch, formatted_train, formatted_test, tokenized_train, tokenized_test
    gc.collect()
    torch.cuda.empty_cache()

log_history = trainer.state.log_history

output_log_file = "training_logs_qwen.txt"


with open(output_log_file, "w") as f:
    for log_entry in log_history:
        f.write(str(log_entry) + "\n")

print(f"Training logs saved to {output_log_file}")