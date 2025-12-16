import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


def main():
    parser = argparse.ArgumentParser(
        description="Push a Hugging Face causal LM checkpoint to the Hub"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the local model checkpoint directory"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g. username/model-name)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="10GB",
        help="Max shard size for HF upload (default: 10GB)"
    )

    args = parser.parse_args()

    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Login to Hugging Face
    hf_token = os.environ.get("HF_API_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_API_TOKEN environment variable is not set")
    login(token=hf_token)

    print(f"Pushing model from {args.model_dir}")
    print(f"Target repo: {args.repo_id}")
    print(f"Dtype: {args.dtype}")

    # Load model (no PEFT)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        fix_mistral_regex=True
    )

    # Push to Hugging Face Hub
    model.push_to_hub(
        args.repo_id,
        safe_serialization=True,
        max_shard_size=args.max_shard_size
    )
    tokenizer.push_to_hub(args.repo_id)

    print("✅ Upload complete!")


if __name__ == "__main__":
    main()
