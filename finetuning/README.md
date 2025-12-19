# LORA Finetuning for Apertus Models

## For Swiss legal dataset

### LORA hyperparameters

We selected our LoRA configuration based on findings from "LoRA Without Regret" (Schulman et al., 2025) and "Practical LoRA Research" (Kirkby & O'Neill, 2025).

    Rank (r=64): The SLDS dataset contains ~60k training examples. Kirkby & O'Neill found that low ranks (r≤8) begin to saturate (stop learning efficiently) around 30k examples. To accommodate the full dataset size and the high entropy of legal reasoning tasks, we selected r=64 to ensure the adapter has sufficient capacity to model the data distribution without bottlenecking.

    Target Modules (All-Linear): Schulman et al. demonstrated that applying LoRA only to attention layers significantly underperforms. We apply adapters to all linear layers (Attn + MLP) to ensure the model matches Full Fine-Tuning performance.

    Learning Rate (2e-4): We adhere to the "10x Rule" proposed by Schulman et al., which observed that the optimal learning rate for LoRA is consistently one order of magnitude higher than that of full fine-tuning for the same model architecture.

    Batch Size (32): To avoid the "large batch penalty" observed in LoRA training dynamics (Schulman et al.), we maintain a moderate effective batch size of 32 rather than scaling up to the hardware limit.


## Notes

- Fixes for NCCL issue involved changing the .sh script and the .toml file!