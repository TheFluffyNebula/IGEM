from datasets import load_dataset
from collections import defaultdict
import json
import os


ds = load_dataset("SetFit/trec-coarse")

labels_in_data = set(example["label"] for split in ["train", "test"] for example in ds[split])
print("Found labels:", sorted(labels_in_data))

# def save_trec_to_json(output_dir="new_src/data/trec"):
#     os.makedirs(output_dir, exist_ok=True)
#     dataset = load_dataset("SetFit/TREC-QC")
    
#     # Manually specify the label names
#     label_names = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]

#     # Collect examples per label
#     per_label = defaultdict(list)
#     for split in ["train", "test"]:
#         for example in dataset[split]:
#             label = example["label"]
#             per_label[label].append({
#                 "question": example["text"],
#                 "choices": label_names,
#                 "answer": label
#             })

#     for label_id, examples in per_label.items():
#         fname = os.path.join(output_dir, f"{label_names[label_id]}.json")
#         with open(fname, "w") as f:
#             json.dump(examples, f, indent=2)
#         print(f"Saved {len(examples)} examples to {fname}")

# save_trec_to_json()
