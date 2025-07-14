from datasets import load_dataset
import json
import os
from collections import defaultdict

output_dir = "new_src/data/agnews"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
ds = load_dataset("SetFit/ag_news")  # returns a dict: {'train': ..., 'test': ...}

# We'll treat the entire dataset as one "subject" for now
for split in ["train", "test"]:
    data = []
    for item in ds[split]:
        question = item["text"]
        label = item["label"]
        
        entry = {
            "question": f"{question.strip()}\nWhich category does this belong to?",
            "choices": ["World", "Sports", "Business", "Sci/Tech"],
            "answer": label
        }
        data.append(entry)
    
    # Save to file
    json_path = os.path.join(output_dir, f"{split}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} examples to {json_path}")
