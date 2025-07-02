from transformers import AutoTokenizer, AutoModelForMultipleChoice

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

save_path = "./new_src/models/gpt2"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
