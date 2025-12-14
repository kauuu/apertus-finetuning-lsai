import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------
# CONFIG — default values (can be overridden via CLI)
# ---------------------------------------------------
DEFAULT_BASE_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"
DEFAULT_MERGED_DIR = "/iopsstor/scratch/cscs/{user}/apertus-finetuning-lsai/merged_70b_model".format(user=os.getenv("USER"))
DEFAULT_DTYPE = torch.float16

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--lora_dir", type=str, required=True, help="Path to LoRA checkpoint directory")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base model name or path")
    parser.add_argument("--merged_dir", type=str, default=DEFAULT_MERGED_DIR, help="Output directory for merged model")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16", help="Torch dtype for model")
    
    args = parser.parse_args()

    DTYPE = torch.float16 if args.dtype == "float16" else torch.bfloat16

    os.makedirs(args.merged_dir, exist_ok=True)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=DTYPE,
        device_map="cpu",
        trust_remote_code=True
    )

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_dir,
        device_map="cpu"
    )

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print("Saving merged model...")
    model.save_pretrained(
        args.merged_dir,
        safe_serialization=True
    )

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, fix_mistral_regex=True)
    tokenizer.save_pretrained(args.merged_dir)

    print("✅ LoRA merge complete")


if __name__ == "__main__":
    main()
