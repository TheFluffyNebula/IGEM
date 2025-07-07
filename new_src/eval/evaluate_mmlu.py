# eval/evaluate_mmlu.py

import os
import json
from models.gpt2_mc import predict_best_choice

def evaluate_mmlu_file(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    correct = 0
    for item in data:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        pred = predict_best_choice(question, choices)
        if pred == answer:
            correct += 1

    total = len(data)
    acc = correct / total
    print(f"{json_path}: {correct}/{total} correct ({acc*100:.2f}%)")
    return acc

def main():
    root = "data/mmlu"
    for file in os.listdir(root):
        if file.endswith(".json"):
            path = os.path.join(root, file)
            evaluate_mmlu_file(path)

if __name__ == "__main__":
    main()
