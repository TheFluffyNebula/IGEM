from datasets import get_dataset_config_names, load_dataset
import json
import os

output_dir = "../data/mmlu"
os.makedirs(output_dir, exist_ok=True)

print("Fetching subject names from 'hendrycks_test'...")
subjects = get_dataset_config_names("hendrycks_test")
# print("Subjects found:", subjects)
# print([(s, type(s)) for s in subjects])

ds = load_dataset("hendrycks_test", "college_physics")
print(ds)

# for subject in subjects:
#     if not subject or subject == "all":
#         continue

#     print(f"Processing subject: {subject}")
#     try:
#         ds = load_dataset("hendrycks_test", subject, split="test")

#         data = []
#         for item in ds:
#             entry = {
#                 "question": item["question"],
#                 "choices": [item["A"], item["B"], item["C"], item["D"]],
#                 "answer": item["answer"],
#             }
#             data.append(entry)

#         json_path = os.path.join(output_dir, f"{subject}.json")
#         with open(json_path, "w") as f:
#             json.dump(data, f, indent=2)

#         print(f"Saved {len(data)} examples to {json_path}")

#     except Exception as e:
#         print(f"Failed to process subject '{subject}': {e}")
