import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
model.eval()

def score_choice(prompt: str, choice: str) -> float:
    input_text = prompt + " " + choice
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        log_likelihood = -loss.item() * input_ids.size(1)

    return log_likelihood

def predict_best_choice(prompt: str, choices: list[str]) -> int:
    scores = [score_choice(prompt, c) for c in choices]
    return int(torch.tensor(scores).argmax().item())
