import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from peft import LoraConfig, get_peft_model

class HuggingFaceWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, packed, **kwargs):
        # packed: [B, 2, L]
        input_ids     = packed[:, 0, :]  # [B, L]
        attention_mask= packed[:, 1, :]  # [B, L]

        return self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits


def get_gpt2_lora(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=4,
    )
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Important for GPT2

    model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if name.startswith("base_model.model.score"):
            param.requires_grad = True
    # Wrap the model before returning
    return HuggingFaceWrapper(model).to(device)
