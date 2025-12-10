from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback, BitsAndBytesConfig
from datasets import Dataset
import torch
import torch.nn.functional as F
import json
import os
import evaluate
import numpy as np
import pandas as pd

import sys
import importlib.util

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

train_df = pd.read_csv('TrainingData/training_data.csv')
test_df = pd.read_csv('TrainingData/test_data.csv')

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def format_conversation_data(examples):
    formatted_texts = []
    
    for messages_json, target in zip(examples['messages'], examples['target']):
        messages = json.loads(messages_json)
        
        conversation = "<|im_start|>system\nYou are Friday, a helpful AI assistant.<|im_end|>\n"
        
        for msg in messages:
            if msg['role'] == 'user':
                conversation += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            else:
                conversation += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        
        conversation += f"<|im_start|>assistant\n{target}<|im_end|>"
        
        formatted_texts.append(conversation)
    
    return {"text": formatted_texts}

metric = evaluate.load('accuracy')

def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
tokenizer.model_max_length = 512
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.resize_token_embeddings(len(tokenizer))

print("Loading reference model for KL divergence...")
reference_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=bnb_config
)
reference_model.resize_token_embeddings(len(tokenizer))
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False

print("Applying LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.config.use_cache = False

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_attention_mask=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

formatted_train = train_dataset.map(format_conversation_data, batched=True, remove_columns=train_dataset.column_names)
formatted_test = test_dataset.map(format_conversation_data, batched=True, remove_columns=test_dataset.column_names)

tokenized_train = formatted_train.map(tokenize_function, batched=True, remove_columns=formatted_train.column_names)
tokenized_test = formatted_test.map(tokenize_function, batched=True, remove_columns=formatted_test.column_names)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)


class KLDivergenceTrainer(Trainer):
    def __init__(self, *args, reference_model=None, kl_coef=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.kl_coef = kl_coef
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if self.reference_model is not None:
            with torch.no_grad():
                ref_outputs = self.reference_model(**inputs)
                ref_logits = ref_outputs.logits
            
            shift_ref_logits = ref_logits[..., :-1, :].contiguous()
            
            model_log_probs = F.log_softmax(shift_logits, dim=-1)
            ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)
            
            kl_div = F.kl_div(
                model_log_probs.view(-1, model_log_probs.size(-1)),
                ref_log_probs.view(-1, ref_log_probs.size(-1)),
                reduction='batchmean',
                log_target=True
            )
            
            loss = lm_loss + self.kl_coef * kl_div
        else:
            loss = lm_loss
        
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="prompt_classifier_Qwen",
    eval_strategy='steps',
    eval_steps=100,
    save_strategy='steps',
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    push_to_hub=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    gradient_accumulation_steps=8,
    report_to=['tensorboard'],
    learning_rate=2e-4,
    lr_scheduler_type="cosine_with_restarts",
    fp16=True,
    max_grad_norm=1.0,
    logging_steps=10,
    save_total_limit=5,
    dataloader_pin_memory=False,
    gradient_checkpointing=False,
    optim="adamw_torch",
    warmup_ratio=0.1,
    weight_decay=0.02,
    greater_is_better=False
)

trainer = KLDivergenceTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    reference_model=reference_model,
    kl_coef=0.125,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
)



trainer.train()

log_history = trainer.state.log_history

output_log_file = "training_logs_qwen.txt"

with open(output_log_file, "w") as f:
    for log_entry in log_history:
        f.write(str(log_entry) + "\n")

print(f"Training logs saved to {output_log_file}")