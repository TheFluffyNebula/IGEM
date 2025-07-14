from datasets import load_dataset
import json
import os

# Load dataset
ds = load_dataset("CogComp/trec")
train = ds["train"].rename_column("coarse_label", "label")
test = ds["test"].rename_column("coarse_label", "label")

# Combine train and test splits
all_data = list(train) + list(test)

# Class label names
label_names = train.features["label"].names  # ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
label_to_index = {label: idx for idx, label in enumerate(label_names)}

# Group by class (for 6 JSONs, 1 per class)
output_dir = "new_src/data/trec_coarse"
os.makedirs(output_dir, exist_ok=True)

examples_by_class = {label: [] for label in label_names}

for item in all_data:
    question_text = item["text"]
    true_label = label_names[item["label"]]
    answer_idx = label_to_index[true_label]

    example = {
        "question": question_text,
        "choices": label_names,  # list of all 6 labels
        "answer": answer_idx     # index into the choices list
    }

    examples_by_class[true_label].append(example)

# Save one file per class (to simulate tasks)
for label, examples in examples_by_class.items():
    filename = os.path.join(output_dir, f"{label}.json")
    with open(filename, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Wrote {len(examples)} examples to {filename}")
