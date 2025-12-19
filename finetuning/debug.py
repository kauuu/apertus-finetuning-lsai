import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
from textwrap import dedent
import os
import sys
from datetime import datetime
from accelerate import Accelerator
from math import ceil
from bert_score import score

# ========================
# CLI ARGUMENTS
# ========================
parser = argparse.ArgumentParser(description="SLDS Headnote Generation (Cross-Lingual Support)")

# Model Configuration
parser.add_argument(
    "--model-path", 
    type=str, 
    default="swiss-ai/Apertus-8B-Instruct-2509", 
    help="HuggingFace model path or local directory."
)

# Inference Configuration
parser.add_argument("--one-shot", action="store_true", help="Enable one-shot prompting using matched language pairs.")
parser.add_argument("--samples", type=int, default=25, help="Number of test samples to process (default: 25).")
parser.add_argument("--batch-size", type=int, default=5, help="Batch size for inference (default: 5).")

# Language Filtering (Crucial for Rescaled BERTScore)
parser.add_argument("--decision-lang", type=str, default=None, help="Filter by decision language (e.g., 'de', 'fr', 'it').")
parser.add_argument("--headnote-lang", type=str, default=None, help="Filter by target headnote language (e.g., 'de', 'fr', 'it').")

args = parser.parse_args()

# ========================
# CONFIGURATION
# ========================
MODEL_PATH = args.model_path 
DATASET_NAME = "ipst/slds"
MAX_LENGTH = 512              # Max new tokens to generate

USE_ONE_SHOT = args.one_shot
SAMPLE_SIZE = args.samples
BATCH_SIZE = args.batch_size
FILTER_DEC_LANG = args.decision_lang
FILTER_HEAD_LANG = args.headnote_lang

# Logging setup
OUTPUT_DIR = "debug_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
mode_label = "1shot" if USE_ONE_SHOT else "0shot"
filter_label = ""
if FILTER_DEC_LANG: filter_label += f"_{FILTER_DEC_LANG}"
if FILTER_HEAD_LANG: filter_label += f"2{FILTER_HEAD_LANG}"

safe_model_name = MODEL_PATH.replace("/", "_")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"slds_{safe_model_name}_{mode_label}{filter_label}_{timestamp}.txt")

# Language Mapping for Prompting
LANG_NAME_MAP = {
    'de': 'German',
    'fr': 'French',
    'it': 'Italian',
    'en': 'English',
    'rm': 'Romansh'
}

# ========================
# PROMPT TEMPLATES
# ========================
BASE_SYSTEM_PROMPT = dedent("""
You are a legal expert specializing in Swiss Federal Supreme Court decisions with extensive knowledge of legal terminology and conventions in German, French, and Italian. Your task is to generate a headnote for a provided leading decision. A headnote is a concise summary that captures the key legal points and significance of the decision. It is not merely a summary of the content but highlights the aspects that make the decision "leading" and important for future legislation.

When generating the headnote:

1. Focus on the core legal reasoning and key considerations that establish the decision's significance.
2. Include any relevant references to legal articles (prefixed with "Art.") and considerations (prefixed with "E." in German or "consid." in French/Italian).
3. Use precise legal terminology and adhere to the formal and professional style typical of Swiss Federal Supreme Court headnotes.
4. Ensure clarity and coherence, so the headnote is logically structured and easy to understand in the specified language.

Your response should consist solely of the headnote in the language specified by the user prompt.
""")

ONE_SHOT_TEMPLATE = dedent("""
Example Decision:
{example_decision}

Example Headnote:
{example_headnote}
""")

TASK_TEMPLATE = dedent("""
Decision:
{target_decision}

As the legal expect, you are asked to generate the headnote in {target_language}!

Headnote:
""")

# ========================
# HELPER: GET ONE-SHOT EXAMPLES (CROSS-LINGUAL)
# ========================
def prepare_one_shot_examples(dataset_split):
    """
    Pre-fetches one example per language pair.
    """
    shots = {}
    languages = ['de', 'fr', 'it']
    
    # Shuffle once
    shuffled = dataset_split.shuffle(seed=42)
    
    for dec_lang in languages:
        for head_lang in languages:
            try:
                subset = shuffled.filter(
                    lambda x: x.get('decision_language') == dec_lang and x.get('headnote_language') == head_lang, 
                    load_from_cache_file=False
                )
                if len(subset) > 0:
                    shots[(dec_lang, head_lang)] = subset[0]
            except Exception:
                continue
            
    # Generic fallback
    if len(dataset_split) > 0:
        shots['default'] = dataset_split[0]
    return shots

def construct_prompt(target_sample, shots):
    """
    Builds the prompt with context and instructions.
    """
    prompt = BASE_SYSTEM_PROMPT
    
    # 1. Handle One-Shot Context
    if USE_ONE_SHOT:
        d_lang = target_sample.get('decision_language')
        h_lang = target_sample.get('headnote_language')
        
        # Priority: Exact Pair -> Decision Lang Match -> Default
        shot = shots.get((d_lang, h_lang))
        if not shot and d_lang:
            for key in shots:
                if isinstance(key, tuple) and key[0] == d_lang:
                    shot = shots[key]
                    break
        if not shot:
            shot = shots.get('default')
            
        if shot:
            prompt += ONE_SHOT_TEMPLATE.format(
                example_decision=shot['decision'],
                example_headnote=shot['headnote']
            )
    
    # 2. Resolve Target Language Name
    target_lang_code = target_sample.get('headnote_language', 'de')
    target_lang_name = LANG_NAME_MAP.get(target_lang_code, "German")
    
    # 3. Add the Task
    prompt += TASK_TEMPLATE.format(
        target_decision=target_sample['decision'],
        target_language=target_lang_name
    )
    return prompt

# ========================
# MAIN EXECUTION
# ========================
accelerator = Accelerator()

if accelerator.is_main_process:
    # Print Config
    print(f"--- Configuration ---", flush=True)
    print(f"Model Path: {MODEL_PATH}", flush=True)
    print(f"Mode: {'One-Shot' if USE_ONE_SHOT else 'Zero-Shot'}", flush=True)
    print(f"Filters: Dec={FILTER_DEC_LANG or 'Any'}, Head={FILTER_HEAD_LANG or 'Any'}", flush=True)
    print(f"Samples: {SAMPLE_SIZE} | Batch: {BATCH_SIZE}", flush=True)
    print(f"Output: {OUTPUT_FILE}", flush=True)
    print(f"---------------------", flush=True)

    # 1. Load Data
    print(f"Loading dataset {DATASET_NAME}...", flush=True)
    dataset_test = load_dataset(DATASET_NAME, split="test")
    dataset_train = load_dataset(DATASET_NAME, split="train")
    
    # Apply Filters
    if FILTER_DEC_LANG:
        dataset_test = dataset_test.filter(lambda x: x['decision_language'] == FILTER_DEC_LANG)
    if FILTER_HEAD_LANG:
        dataset_test = dataset_test.filter(lambda x: x['headnote_language'] == FILTER_HEAD_LANG)
    
    available_samples = len(dataset_test)
    if available_samples == 0:
        print("Error: No samples found matching the language criteria!", flush=True)
        sys.exit(1)
        
    final_sample_count = min(SAMPLE_SIZE, available_samples)
    samples = dataset_test.select(range(final_sample_count))
    print(f"Processing {final_sample_count} samples.", flush=True)
    
    # Prepare shots
    shots_map = {}
    if USE_ONE_SHOT:
        print("Preparing cross-lingual one-shot examples...", flush=True)
        shots_map = prepare_one_shot_examples(dataset_train)

    # 2. Load Model & Tokenizer
    print(f"Loading model {MODEL_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16
    )
    model.eval()
    
    generation_config = GenerationConfig(
        max_new_tokens=MAX_LENGTH,
        do_sample=False, 
        temperature=1.0, 
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # ========================
    # INFERENCE LOOP
    # ========================
    print(f"Starting inference...", flush=True)
    
    num_batches = ceil(final_sample_count / BATCH_SIZE)
    all_headnotes = []
    all_inputs = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, final_sample_count)
        batch_indices = range(start_idx, end_idx)
        batch_samples = samples.select(batch_indices)
        
        # Build Prompts
        input_texts = [construct_prompt(s, shots_map) for s in batch_samples]
        all_inputs.extend(input_texts)

        # Tokenize & Generate
        tokenizer.truncation_side = "right"
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=4096, 
            padding=True
        )
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # Slice Output
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        batch_headnotes = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        all_headnotes.extend(batch_headnotes)
        print(f"Batch {batch_idx+1}/{num_batches} processed.", flush=True)

    # ========================
    # EVALUATION: ACADEMIC BERTSCORE
    # ========================
    print("Inference complete. Calculating BERTScore...", flush=True)
    torch.cuda.empty_cache()

    references = [s['headnote'] for s in samples]
    
    # Matching Logic from your Reference Script
    use_rescaling = False
    bert_lang = None
    
    if FILTER_HEAD_LANG:
        use_rescaling = True
        bert_lang = FILTER_HEAD_LANG
        
        # Handle Romansh edge case (matches reference script logic)
        if bert_lang == "rm":
            print("⚠️ Romansh detected. Using Italian baseline for BERTScore (as per standard).", flush=True)
            bert_lang = "it"
            
        print(f"✅ Rescaling enabled for language: {bert_lang}", flush=True)
    else:
        print("⚠️ No specific headnote language filtered. Using RAW scores (not comparable to baselines).", flush=True)

    # Calculate
    P, R, F1 = score(
        all_headnotes, 
        references, 
        model_type="bert-base-multilingual-cased", 
        num_layers=9, 
        verbose=True,
        device=accelerator.device,
        lang=bert_lang,                    # Pass language if available
        rescale_with_baseline=use_rescaling # Enable rescaling if lang is set
    )

    avg_f1 = F1.mean().item()
    avg_P = P.mean().item()
    avg_R = R.mean().item()

    print(f"Scores -> P: {avg_P:.4f} | R: {avg_R:.4f} | F1: {avg_f1:.4f}", flush=True)

    # ========================
    # SAVE LOGS
    # ========================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"SLDS Generation Log - {datetime.now()}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Mode: {'1-SHOT' if USE_ONE_SHOT else '0-SHOT'}\n")
        f.write(f"Metric: {'Rescaled' if use_rescaling else 'Raw'} BERTScore\n\n")
        
        f.write("=" * 40 + "\n")
        f.write("AGGREGATE METRICS\n")
        f.write("=" * 40 + "\n")
        f.write(f"F1 (Avg):        {avg_f1:.4f}\n")
        f.write(f"Precision (Avg): {avg_P:.4f}\n")
        f.write(f"Recall (Avg):    {avg_R:.4f}\n\n")

        for i, sample in enumerate(samples):
            f.write(f"--- Sample {i+1} ---\n")
            d_lang = sample.get('decision_language', 'N/A')
            h_lang = sample.get('headnote_language', 'N/A')
            f.write(f"Direction: {d_lang} -> {h_lang}\n")
            
            # Continuous text block
            full_sequence = all_inputs[i] + all_headnotes[i]
            f.write(f"\n[Model Input & Generation]:\n{full_sequence}\n")
            
            f.write(f"\n[Expected Headnote]:\n{sample['headnote']}\n")
            f.write(f"\n> Sample F1: {F1[i].item():.4f}\n")
            f.write("-" * 40 + "\n")

    print(f"Done. Results saved to {OUTPUT_FILE}", flush=True)