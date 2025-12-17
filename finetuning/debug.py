import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
from textwrap import dedent
import os
from datetime import datetime
from accelerate import Accelerator
from math import ceil

# ========================
# CONFIGURATION
# ========================
MODEL_PATH = "kkaushik02/apertus-8b-instruct-full-finetuned"
DATASET_NAME = "ipst/slds"
SAMPLE_SIZE = 10        # total number of samples to process
BATCH_SIZE = 5           # number of samples per batch
MAX_LENGTH = 512

# Make directory to store logs
OUTPUT_DIR = "debug_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create the file path (using a filename-safe timestamp)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"slds_headnotes_output_{timestamp}.txt")

# ========================
# SYSTEM PROMPT
# ========================
SLDS_GENERATION_SYSTEM_PROMPT = dedent("""
You are a legal expert specializing in Swiss Federal Supreme Court decisions with extensive knowledge of legal terminology and conventions in German, French, and Italian. 

Your task is to generate a **headnote** for a provided leading decision. A headnote is a concise summary that captures the key legal points and significance of the decision. It is **not merely a summary**; highlight the aspects that make the decision "leading" and important for future legislation.

When generating the headnote:

1. Focus on the core **legal reasoning** and key considerations that establish the decision's significance.
2. Include **any relevant references to legal articles** (prefixed with "Art.") and considerations (prefixed with "E." in German or "consid." in French/Italian).
3. Use **precise legal terminology** and adhere to the **formal and professional style** typical of Swiss Federal Supreme Court headnotes.
4. Ensure clarity and coherence; the headnote should be logically structured and easy to understand.

Output format:

- The headnote should be in the same language as the decision unless specified otherwise.
- The output should consist **solely of the headnote**, no extra commentary or explanations.

Example usage:

Decision: {{decision_text}}\n
Headnote:
""")

# ========================
# LOAD MODEL & TOKENIZER
# ========================
accelerator = Accelerator()
if accelerator.is_main_process:
    dataset = load_dataset(DATASET_NAME, split="test")
    samples = dataset.select(range(SAMPLE_SIZE))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # FIX 1: Set Padding Side to LEFT for generation
    tokenizer.padding_side = "left" 
    
    # Ensure pad token is set (Llama-based models often lack a default pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading model {MODEL_PATH} across GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16
    )
    model.eval()
    
    generation_config = GenerationConfig(
        max_new_tokens=MAX_LENGTH,
        do_sample=False,
        temperature=1.0, # Temperature is ignored if do_sample=False, but good practice
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # ========================
    # INFERENCE LOOP IN BATCHES
    # ========================
    num_batches = ceil(SAMPLE_SIZE / BATCH_SIZE)
    all_headnotes = []

    for batch_idx in range(num_batches):
        batch_indices = list(range(batch_idx*BATCH_SIZE, (batch_idx+1)*BATCH_SIZE))
        batch_samples = samples.select(batch_indices)
        
        input_texts = [
            SLDS_GENERATION_SYSTEM_PROMPT.replace("{{decision_text}}", s["decision"])
            for s in batch_samples
        ]

        # FIX 2: Handle Truncation carefully
        # Note: If input > model_max_length, "left" removes instructions, "right" removes "Headnote:"
        # ideally, you should truncate s["decision"] before putting it in the prompt.
        # For now, we revert to default (right) to ensure instructions are kept, 
        # but be aware lengthy decisions might lose the "Headnote:" trigger at the end.
        tokenizer.truncation_side = "right" 

        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=4096, # Set a safe hard limit if model_max_length is huge
            padding=True
        )
        
        first_device = list(model.hf_device_map.values())[0]
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # FIX 3: Slice the output to keep ONLY the generated tokens
        # We calculate the length of the input tokens
        input_length = inputs["input_ids"].shape[1]
        
        # We slice the output tensor from [input_length] to the end
        generated_tokens = outputs[:, input_length:]
        
        # Decode only the new tokens
        headnotes = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        all_headnotes.extend(headnotes)
        
        # Optional: Print progress
        print(f"Batch {batch_idx+1}/{num_batches} processed.")

    # ========================
    # SAVE RESULTS
    # ========================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"SLDS Headnote Generation Log for Model {MODEL_PATH} - {datetime.now()}\n\n")
        for i, sample in enumerate(samples):
            f.write(f"--- Sample {i+1} ---\n")
            f.write(f"Generated Headnote:\n{all_headnotes[i]}\n\n")
            f.write(f"Expected Headnote:\n{sample['headnote']}")
            f.write("-" * 20 + "\n") # Separator