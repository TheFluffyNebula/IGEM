from torch.utils.data import Dataset
from avalanche.benchmarks import NCScenario
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from transformers import GPT2Tokenizer
import os
import json
import random
import numpy as np

class MMLUDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Using setattr instead of direct assignment

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
        # 2. MUST return a tuple (input, target)
        return input_ids, label  # Not a dictionary!

def make_mmlu_benchmark(mmlu_root: str, n_experiences: int, seed: int):
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

        # Split data
        random.shuffle(data)
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        # Create base datasets
        train_base = MMLUDataset(train_data, tokenizer)
        test_base = MMLUDataset(test_data, tokenizer)

        # Set targets RIGHT BEFORE wrapping
        train_base.targets = [entry["answer"] for entry in train_data]
        test_base.targets = [entry["answer"] for entry in test_data]

        # Now wrap with Avalanche
        av_train = AvalancheDataset(
            ClassificationDataset(train_base),
            task_labels=task_id
        )
        av_test = AvalancheDataset(
            ClassificationDataset(test_base),
            task_labels=task_id
        )

        train_datasets.append(av_train)
        test_datasets.append(av_test)
        task_labels.append(task_id)

    scenario = NCScenario(
        train_datasets,
        test_datasets,
        n_experiences=n_experiences,
        task_labels=True,
        shuffle=False,
        seed=seed
    )
    return scenario

if __name__ == '__main__':
    # print("hi")
    benchmark = make_mmlu_benchmark(
        mmlu_root="new_src/data/mmlu",
        n_experiences=5,  # Number of subjects/tasks
        seed=42
    )

    for experience in benchmark.train_stream:
        print(f"Experience {experience.current_experience}")
        print(f"Classes: {experience.classes_in_this_experience}")
        print(f"Dataset size: {len(experience.dataset)}")
