import os
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or the specific GPU you want
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_P2P_DISABLE"] = "1"  # <- workaround
os.environ["NCCL_IB_DISABLE"] = "1"

# 1) A tiny Dataset wrapper for one JSON file’s train/test split
class MMLUSingleTask(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.encodings = tokenizer(
            [ entry["question"] + "\n" +
              "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(entry["choices"]))
              for entry in data ],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        self.labels = [ entry["answer"] for entry in data ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = { k: torch.tensor(v[i]) for k, v in self.encodings.items() }
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

# 2) Load one MMLU subject (JSON) and split
root = "new_src/data/mmlu"
fname = sorted(f for f in os.listdir(root) if f.endswith(".json"))[0]
data = json.load(open(os.path.join(root, fname)))
split = int(0.8 * len(data))
train_data, test_data = data[:split], data[split:]

# 3) Shared tokenizer → add [PAD]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.pad_token = "[PAD]"

# 4) Build datasets
train_ds = MMLUSingleTask(train_data, tokenizer)
eval_ds  = MMLUSingleTask(test_data,  tokenizer)

# 5) Model + LoRA adapters
from peft import LoraConfig, get_peft_model
base = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=6)

# --- WITHOUT LORA ---
model = base
# --- WITH LORA ---
# lora_cfg = LoraConfig(
#     r=16, lora_alpha=32, lora_dropout=0.05,
#     target_modules=["c_attn","c_proj"], bias="none", task_type="SEQ_CLS"
# )
# model = get_peft_model(base, lora_cfg)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
# unfreeze the classification head
for n,p in model.named_parameters():
    
    if n.startswith("score."):
        p.requires_grad = True

# 6) TrainingArguments for a quick sanity run
training_args = TrainingArguments(
    output_dir="tmp_mmlu",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=3e-4,
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    disable_tqdm=False,
    report_to="none",
)
import pandas as pd
# 7) compute_metrics callback
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    acc= {"accuracy_base": accuracy_score(p.label_ids, preds)}
    df = pd.read_csv("gpt2_mmlu_test_results.csv", )
    df = pd.concat([df, pd.DataFrame(acc, index=[1])])
    df.to_csv("gpt2_mmlu_test_results.csv")

# 8) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 9) Run train & eval
trainer.train()
print(trainer.evaluate())
