import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

from peft import LoraConfig, get_peft_model

def get_gpt2_lora(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=4,
    ).to(device)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

    model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        # rank of the adapter
        r=8,
        # scaling factor
        lora_alpha=32,
        # dropout on the adapter
        lora_dropout=0.05,
        # which submodules to adapt (GPT2’s attention proj layers)
        target_modules=["c_attn", "c_proj"],
        # no extra bias params
        bias="none",
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)
    return model.to(device)
