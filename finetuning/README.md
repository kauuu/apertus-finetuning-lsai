# Our README for LORA Finetuning for Apertus Models

## For Swiss legal dataset

### LORA hyperparameters

We selected our LoRA configuration based on findings from "LoRA Without Regret" (Schulman et al., 2025) and "Practical LoRA Research" (Kirkby & O'Neill, 2025).

    Rank (r=64): The SLDS dataset contains ~60k training examples. Kirkby & O'Neill found that low ranks (r≤8) begin to saturate (stop learning efficiently) around 30k examples. To accommodate the full dataset size and the high entropy of legal reasoning tasks, we selected r=64 to ensure the adapter has sufficient capacity to model the data distribution without bottlenecking.

    Target Modules (All-Linear): Schulman et al. demonstrated that applying LoRA only to attention layers significantly underperforms. We apply adapters to all linear layers (Attn + MLP) to ensure the model matches Full Fine-Tuning performance.

    Learning Rate (2e-4): We adhere to the "10x Rule" proposed by Schulman et al., which observed that the optimal learning rate for LoRA is consistently one order of magnitude higher than that of full fine-tuning for the same model architecture.

    Batch Size (32): To avoid the "large batch penalty" observed in LoRA training dynamics (Schulman et al.), we maintain a moderate effective batch size of 32 rather than scaling up to the hardware limit.


# Original README for Apertus Fine-Tuning Recipes

This repository provides fine-tuning recipes for Swiss AI’s Apertus language models (8B and 70B), supporting both full-parameter and LoRA-based approaches.
Built on top of popular frameworks including TRL, Accelerate, and Transformers, the recipes are optimized for efficient training on modern GPUs.
LoRA fine-tuning of the 8B model can be done on a single 40 GB GPU, while training the 70B model requires a multi-GPU setup.


## 🔗 Resources
- [Apertus 8B Instruct](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509)  
- [Apertus 70B Instruct](https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509)  
- [Full collection on HF](https://huggingface.co/collections/swiss-ai/apertus-llm-68b699e65415c231ace3b059)  

---

## ⚡ Quickstart

```bash
# 1. Create and activate environment
uv venv apertus --python 3.10 && source apertus/bin/activate

# 2. Install PyTorch (CUDA 12.8 wheels)
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu128

# 3. Install project requirements
uv pip install -r requirements.txt

# 4. Launch LoRA training on a single GPU
python sft_train.py --config configs/sft_lora.yaml
````

---

## Model Selection

All scripts work with both 8B and 70B versions. Switch model size by editing `model_path`:

```python
# Default: 8B
model_path = "swiss-ai/Apertus-8B-Instruct-2509"

# To use 70B:
# model_path = "swiss-ai/Apertus-70B-Instruct-2509"
```

Device mapping and configuration are handled automatically.

---

## Fine-Tuning

### Full-parameter training (4 GPUs)

```bash
# Standard attention
accelerate launch --config_file configs/zero3.yaml sft_train.py --config configs/sft_full.yaml

# With FlashAttention3
accelerate launch --config_file configs/zero3.yaml sft_train.py \
    --config configs/sft_full.yaml \
    --attn_implementation kernels-community/vllm-flash-attn3
```

### LoRA training (1 GPU)

```bash
python sft_train.py --config configs/sft_lora.yaml
```

---

### Multi-Node training (3 nodes x 4 GPUs)

```bash
# Standard attention
bash --nodes=3 submit_multinode.sh
```
## Customization

You can adjust datasets and hyperparameters either by editing the config YAMLs (`sft_lora.yaml`, `sft_full.yaml`) or passing overrides directly:

```bash
accelerate launch --config_file configs/zero3.yaml \
    sft_train.py --config configs/sft_full.yaml \
    --dataset_name YOUR_DATASET
```

---

## Model Saving


After training completes, your fine-tuned models are saved in the following locations:

- **LoRA Training**: `Apertus-FT/output/apertus_lora/`
- **Full Fine-tuning**: `Apertus-FT/output/apertus_full/`

Each output directory contains:
- `adapter_model.safetensors` (LoRA only) - The LoRA adapter weights
- `adapter_config.json` (LoRA only) - LoRA configuration
- `training_args.bin` - Training arguments used
- `trainer_state.json` - Training state and metrics
- `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
- `config.json` - Model configuration

---

## Using Your Fine-tuned Models

#### For LoRA Adapters

LoRA adapters are lightweight and can be loaded with the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("swiss-ai/Apertus-8B-Instruct-2509")
tokenizer = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-Instruct-2509")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "Apertus-FT/output/apertus_lora/")

# For inference, you can merge the adapter (optional)
model = model.merge_and_unload()
```

#### For Full Fine-tuned Models

Full fine-tuned models can be loaded directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("Apertus-FT/output/apertus_full/")
tokenizer = AutoTokenizer.from_pretrained("Apertus-FT/output/apertus_full/")
```

---

## Contributors

- [Kaustubh Ponkshe](https://kaustubhp11.github.io/)
- [Raghav Singhal](https://raghavsinghal10.github.io/)
