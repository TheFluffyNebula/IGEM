import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from peft import LoraConfig, get_peft_model
from util import get_tokenizer
class HuggingFaceWrapper(nn.Module):
    def __init__(self, peft_model, tokenizer):
        super().__init__()
        self.model = peft_model
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, input_ids):
        # build mask internally
        attention_mask = (input_ids != self.pad_token_id).long()
        #print(f"Attention mask sum: {attention_mask.sum(dim=1)}" )
        logits= self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits
        #print(f"Logits Shape: {logits.shape}")
        
        return logits



def get_gpt2_lora(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=6,
    )
    
    tokenizer = get_tokenizer()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.attn_pdrop = 0.0
    model.config.resid_pdrop = 0.0
    for name, param in model.named_parameters():
        if name.startswith("base_model.model.score"):
            param.requires_grad = True
    layers_to_keep = { 10,11}

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            layer_idx = int(name.split(".")[4])
            if layer_idx not in layers_to_keep:
                p.requires_grad = False
        if p.requires_grad:
            print(name)
    cnt = sum([p.numel() for _, p in model.named_parameters() if p.requires_grad])
    n_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    n_scalars = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable tensors:", n_tensors)
    print("Trainable scalars:", n_scalars)
    # Wrap the model before returning
    return HuggingFaceWrapper(model,tokenizer).to(device)
