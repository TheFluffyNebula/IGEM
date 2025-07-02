# eval/mmlu_benchmark.py

import os
import json
from torch.utils.data import Dataset
from avalanche.benchmarks import NCScenario
from avalanche.benchmarks.utils import AvalancheDataset
from transformers import GPT2Tokenizer


class MMLUDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        prompt = entry["question"] + "\n" + "\n".join(
            [f"{chr(65+i)}. {choice}" for i, choice in enumerate(entry["choices"])]
        )
        input_ids = self.tokenizer(
            prompt, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )["input_ids"].squeeze(0)
        label = entry["answer"]  # 0-based index
        return input_ids, label


def make_mmlu_benchmark(mmlu_root: str, n_experiences: int, seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    all_files = sorted(f for f in os.listdir(mmlu_root) if f.endswith(".json"))

    if len(all_files) < n_experiences:
        raise ValueError(f"Requested {n_experiences} tasks but found only {len(all_files)} subjects.")

    datasets = []
    for task_id, fname in enumerate(all_files[:n_experiences]):
        with open(os.path.join(mmlu_root, fname)) as f:
            data = json.load(f)
        dataset = MMLUDataset(data, tokenizer)
        dataset = AvalancheDataset(dataset, task_labels=task_id)
        datasets.append(dataset)

    return NCScenario(datasets)
