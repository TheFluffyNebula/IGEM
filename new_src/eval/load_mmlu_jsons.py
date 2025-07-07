from datasets import get_dataset_config_names, load_dataset
import json
import os
import time

output_dir = "new_src/data/mmlu"
os.makedirs(output_dir, exist_ok=True)

# print("Fetching subject names from 'hendrycks_test'...")
subjects = get_dataset_config_names("cais/mmlu")
# print("Subjects found:", subjects)

# ds = load_dataset("cais/mmlu")
# print(ds)

for subject in subjects:
    time.sleep(0.1)
    if not subject or subject == "all":
        continue

    print(f"Processing subject: {subject}")
    try:
        ds = load_dataset("cais/mmlu", subject, split="test")

        data = []
        for item in ds:
            entry = {
                "question": item["question"],
                "choices": item["choices"],  # already a list
                "answer": item["answer"],    # integer index: 0, 1, 2, or 3
            }
            data.append(entry)

        json_path = os.path.join(output_dir, f"{subject}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(data)} examples to {json_path}")

    except Exception as e:
        print(f"Failed to process subject '{subject}': {e}")
