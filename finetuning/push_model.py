from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import os

# Login to Hugging Face
login(token=os.environ["HF_API_TOKEN"])

# Paths
MODEL_DIR = "/iopsstor/scratch/cscs/{user}/apertus-finetuning-lsai/merged_70b_model".format(user=os.getenv("USER"))

REPO_ID = "kkaushik02/apertus-70b-instruct-lora-merged"

# Load merged model (no PEFT)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    fix_mistral_regex=True
)


# Push to Hugging Face Hub
model.push_to_hub(
    REPO_ID,
    safe_serialization=True,
    max_shard_size="10GB"
)

tokenizer.push_to_hub(REPO_ID)
