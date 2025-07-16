from torch.utils.data import Dataset
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks import GenericCLScenario, task_incremental_benchmark, with_task_labels, benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset, _make_taskaware_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from transformers import GPT2Tokenizer
import os
import json
import random
import torch
import numpy as np
import util
def make_task_dataset(data, tokenizer, task_id, max_length=512):
    input_ids_list, mask_list, labels = [], [], []
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"  

    for entry in data:
        enc = tokenizer(
            entry["question"] + "\n" +
            "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(entry["choices"])) + tokenizer.eos_token,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids_list.append(enc["input_ids"].squeeze(0))
        mask_list.append(enc["attention_mask"].squeeze(0))
        labels.append(entry["answer"])  # already 0–3
        # print(f"\n[DEBUG] Sample Prompt:\n{entry['question']}")
        # print(f"Choices: {entry['choices']}")
        # print(f"Answer Label: {entry['answer']}")
        # print(f"Input IDs shape: {enc['input_ids'].shape}, Attention mask shape: {enc['attention_mask'].shape}")

    # Stack everything
    X_ids  = torch.stack(input_ids_list, dim=0)    # [N, L]
    X_mask = torch.stack(mask_list,       dim=0)   # [N, L]
    # Pack into one Tensor: [N, 2, L]
    X = torch.stack([X_ids, X_mask], dim=1)       

    y = torch.tensor(labels, dtype=torch.long)    # [N]
    t = torch.full((len(labels),), task_id, dtype=torch.long)

    # Avalanche will see samples as (X[i], y[i], t[i])
    #return AvalancheDataset(TensorDataset(X, y))
    return AvalancheDataset(TensorDataset(X_ids, y))

        
def make_mmlu_benchmark(mmlu_root: str,n_experiences, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tokenizer = util.get_tokenizer()
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = "[PAD]"
    all_files = sorted(f for f in os.listdir(mmlu_root) if f.endswith(".json"))

    if len(all_files) < n_experiences:
        raise ValueError(f"Requested {n_experiences} tasks, but only found {len(all_files)} subjects.")

    train_datasets = []
    test_datasets = []
    task_labels = []

    all_files = all_files[::-1]

    for task_id, fname in enumerate(all_files[:n_experiences]):
        with open(os.path.join(mmlu_root, fname)) as f:
            data = json.load(f)
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        # print("Data LEngth:", len(data))
        train_dataset = make_task_dataset(train_data, tokenizer, task_id)
        test_dataset =  make_task_dataset(test_data, tokenizer, task_id)
        # print("Train and test len:", len(train_dataset), len(test_dataset))
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    base_bench = benchmark_from_datasets(
        train=train_datasets,             # List[AvalancheDataset]
        test=test_datasets,              # List[AvalancheDataset]
    )

    return base_bench
