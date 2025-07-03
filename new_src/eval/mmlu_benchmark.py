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

def make_task_dataset(data, tokenizer, task_id, max_length=512):
    """
    returns: X, y, t
    """
    X = []
    y = []
    t = [task_id] * len(data)
    for entry in data:
        prompt = entry["question"] + "\n" + "\n".join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(entry["choices"])]
        )
        input_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            )["input_ids"].squeeze(0)
        label = entry["answer"]
        X.append(input_ids)
        y.append(label)
    X = torch.stack(X, dim=0)
    y = torch.tensor(y)
    # print("shapes:", X.shape, y.shape)
    dataset = TensorDataset(X, y)
    return _make_taskaware_classification_dataset(dataset, task_labels=t)
        
class MMLUDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = [entry["answer"] for entry in data]  # Must be direct attribute

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        prompt = entry["question"] + "\n" + "\n".join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(entry["choices"])]
        )
        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        label = entry["answer"]
        return input_ids, label  # Must return (input, target) tuple

def make_mmlu_benchmark(mmlu_root: str,n_experiences, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    all_files = sorted(f for f in os.listdir(mmlu_root) if f.endswith(".json"))

    if len(all_files) < n_experiences:
        raise ValueError(f"Requested {n_experiences} tasks, but only found {len(all_files)} subjects.")

    train_datasets = []
    test_datasets = []
    task_labels = []

    for task_id, fname in enumerate(all_files[:n_experiences]):
        with open(os.path.join(mmlu_root, fname)) as f:
            data = json.load(f)
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        train_dataset = make_task_dataset(train_data, tokenizer, task_id)
        test_dataset =  make_task_dataset(test_data, tokenizer, task_id)

        # random.shuffle(data)
        # split_idx = int(0.8 * len(data))
        # train_data = data[:split_idx]
        # test_data = data[split_idx:]

        # # Create base datasets
        # train_base = MMLUDataset(train_data, tokenizer)
        # test_base = MMLUDataset(test_data, tokenizer)

        # # Wrap with ClassificationDataset first
        # classified_train = ClassificationDataset(train_base)
        # classified_test = ClassificationDataset(test_base)

        # # Then wrap with AvalancheDataset - NO task_labels here!
        # av_train = AvalancheDataset(classified_train)
        # av_test = AvalancheDataset(classified_test)

        # # Set task labels through the dataset's attributes
        # av_train.task_labels = [task_id]
        # av_test.task_labels = [task_id]

        # train_datasets.append(av_train)
        # test_datasets.append(av_test)
        # task_labels.append(task_id)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    base_bench = benchmark_from_datasets(
        train=train_datasets,             # List[AvalancheDataset]
        test=test_datasets,              # List[AvalancheDataset]
    )

    benchmark = with_task_labels(
        base_bench,
    )
    return benchmark

if __name__ == '__main__':
    # print("hi")
    benchmark = make_mmlu_benchmark(
        mmlu_root="new_src/data/mmlu",
        seed=42,
        n_experiences=5
    )

    for experience in benchmark.train_stream:
        print(f"Experience {experience.current_experience}")
        print(experience.task_label)
