import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- CONFIGURATION ---
# 1. The original base model you started with
base_model_id = "swiss-ai/Apertus-8B-Instruct-2509"

# 2. Your specific LoRA checkpoint path
lora_path = "/users/kkarthikeyan/scratch/apertus-finetuning-lsai/finetuning/Apertus-FT/output/apertus_lora_r64_ctx8k/checkpoint-1500"

# 3. Where to save the final merged model (On Scratch!)
output_dir = "/users/kkarthikeyan/scratch/apertus-finetuning-lsai/merged_model_8b"
# ---------------------

print(f"Loading base model: {base_model_id}...")
# Load base model in FP16 to save RAM
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu", # Load to CPU first to avoid GPU OOM during merge
    trust_remote_code=True
)

print(f"Loading LoRA adapter from: {lora_path}...")
model = PeftModel.from_pretrained(base_model, lora_path)

print("Merging weights (this may take a minute)...")
model = model.merge_and_unload()

print(f"Saving merged model to: {output_dir}...")
model.save_pretrained(output_dir)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)
