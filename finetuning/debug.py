import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
from textwrap import dedent
import os
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
    help="HuggingFace model path or local directory (default: swiss-ai/Apertus-8B-Instruct-2509)."
)

# Inference Configuration
parser.add_argument("--one-shot", action="store_true", help="Enable one-shot prompting using matched language pairs.")
parser.add_argument("--samples", type=int, default=10, help="Number of test samples to process (default: 10).")
parser.add_argument("--batch-size", type=int, default=5, help="Batch size for inference (default: 5).")

args = parser.parse_args()

# ========================
# CONFIGURATION
# ========================
MODEL_PATH = args.model_path 
DATASET_NAME = "ipst/slds"
MAX_LENGTH = 512              # Max new tokens to generate

# Apply other CLI args
USE_ONE_SHOT = args.one_shot
SAMPLE_SIZE = args.samples
BATCH_SIZE = args.batch_size

# Logging setup
OUTPUT_DIR = "debug_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
mode_label = "1shot" if USE_ONE_SHOT else "0shot"
# Sanitize model name for filename (replace / with _)
safe_model_name = MODEL_PATH.replace("/", "_")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"slds_{safe_model_name}_{mode_label}_{timestamp}.txt")

# Language Mapping for Prompting
LANG_NAME_MAP = {
    'de': 'German',
    'fr': 'French',
    'it': 'Italian',
    'en': 'English'
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

# UPDATED: Now includes explicit language instruction
TASK_TEMPLATE = dedent("""
Decision:
{target_decision}

Please generate the headnote in {target_language}.

Headnote:
""")

# ========================
# HELPER: GET ONE-SHOT EXAMPLES (CROSS-LINGUAL)
# ========================
def prepare_one_shot_examples(dataset_split):
    """
    Pre-fetches one example per language pair (Decision Lang -> Headnote Lang).
    Returns a dictionary keyed by tuple: (decision_language, headnote_language).
    """
    shots = {}
    languages = ['de', 'fr', 'it']
    
    # Shuffle once to get random representative examples
    shuffled = dataset_split.shuffle(seed=42)
    
    # Iterate through all possible language combinations (e.g., DE->DE, DE->FR)
    for dec_lang in languages:
        for head_lang in languages:
            try:
                # Filter for this specific pair
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
    Builds the prompt. Uses (decision_language, headnote_language) to find the perfect shot.
    Injects the target language name into the final instruction.
    """
    prompt = BASE_SYSTEM_PROMPT
    
    # 1. Handle One-Shot Context
    if USE_ONE_SHOT:
        d_lang = target_sample.get('decision_language')
        h_lang = target_sample.get('headnote_language')
        
        # Try Exact Pair Match (e.g., DE -> FR)
        shot = shots.get((d_lang, h_lang))
        
        # Fallback: Match Decision Language only
        if not shot and d_lang:
            for key in shots:
                if isinstance(key, tuple) and key[0] == d_lang:
                    shot = shots[key]
                    break
        
        # Last Resort
        if not shot:
            shot = shots.get('default')
            
        if shot:
            prompt += ONE_SHOT_TEMPLATE.format(
                example_decision=shot['decision'],
                example_headnote=shot['headnote']
            )
    
    # 2. Resolve Target Language Name
    target_lang_code = target_sample.get('headnote_language', 'de') # Default to German if missing
    target_lang_name = LANG_NAME_MAP.get(target_lang_code, "German")
    
    # 3. Add the Task with Language Instruction
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
    print(f"--- Configuration ---")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Mode: {'One-Shot' if USE_ONE_SHOT else 'Zero-Shot'}")
    print(f"Samples: {SAMPLE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"---------------------")

    # 1. Load Data
    print(f"Loading dataset {DATASET_NAME}...")
    dataset_test = load_dataset(DATASET_NAME, split="test")
    dataset_train = load_dataset(DATASET_NAME, split="train")
    
    samples = dataset_test.select(range(SAMPLE_SIZE))
    
    # Prepare shots
    shots_map = {}
    if USE_ONE_SHOT:
        print("Preparing cross-lingual one-shot examples...")
        shots_map = prepare_one_shot_examples(dataset_train)
        print(f"Loaded {len(shots_map)} context examples.")
        print(f"Pairs found: {[k for k in shots_map.keys() if isinstance(k, tuple)]}")

    # 2. Load Model & Tokenizer
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading model {MODEL_PATH}...")
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
    print(f"Starting inference...")
    
    num_batches = ceil(SAMPLE_SIZE / BATCH_SIZE)
    all_headnotes = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, SAMPLE_SIZE)
        batch_indices = range(start_idx, end_idx)
        batch_samples = samples.select(batch_indices)
        
        # Build Prompts
        input_texts = [
            construct_prompt(s, shots_map) for s in batch_samples
        ]

        # Tokenize
        tokenizer.truncation_side = "right"
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=4096, 
            padding=True
        )
        
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # Slice Output
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        batch_headnotes = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        all_headnotes.extend(batch_headnotes)
        print(f"Batch {batch_idx+1}/{num_batches} processed.")

    # ========================
    # EVALUATION: BERTSCORE
    # ========================
    print("Inference complete. Calculating BERTScore...")
    torch.cuda.empty_cache()

    references = [s['headnote'] for s in samples]

    P, R, F1 = score(
        all_headnotes, 
        references, 
        model_type="bert-base-multilingual-cased", 
        num_layers=9, 
        verbose=True,
        device=accelerator.device 
    )

    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    print(f"Scores -> P: {avg_precision:.4f} | R: {avg_recall:.4f} | F1: {avg_f1:.4f}")

    # ========================
    # SAVE LOGS
    # ========================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"SLDS Generation Log ({'1-SHOT' if USE_ONE_SHOT else '0-SHOT'}) - {datetime.now()}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Context Strategy: Match (Decision Lang -> Headnote Lang)\n\n")
        
        f.write("=" * 40 + "\n")
        f.write("AGGREGATE METRICS (BERTScore)\n")
        f.write("=" * 40 + "\n")
        f.write(f"F1 (Avg):        {avg_f1:.4f}\n")
        f.write(f"Precision (Avg): {avg_precision:.4f}\n")
        f.write(f"Recall (Avg):    {avg_recall:.4f}\n\n")

        for i, sample in enumerate(samples):
            f.write(f"--- Sample {i+1} ---\n")
            d_lang = sample.get('decision_language', 'N/A')
            h_lang = sample.get('headnote_language', 'N/A')
            f.write(f"Direction: {d_lang} -> {h_lang}\n")
            
            f.write(f"\n[Generated Headnote]:\n{all_headnotes[i]}\n")
            f.write(f"\n[Expected Headnote]:\n{sample['headnote']}\n")
            f.write(f"\n> Sample F1 Score: {F1[i].item():.4f}\n")
            f.write("-" * 40 + "\n")

    print(f"Done. Results saved to {OUTPUT_FILE}")