import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

from transformers.trainer_utils import get_last_checkpoint

from formatting_utils import formatting_prompts_func


def main(script_args, training_args, model_args):
    # ------------------------
    # 1. Load model & tokenizer
    # ------------------------
    store_base_dir = "./"
    # Ensure checkpoint dir exists
    os.makedirs(training_args.output_dir, exist_ok=True)

    print("\n=== CHECKPOINT CONFIGURATION ===")
    print("Output dir:", training_args.output_dir)
    print("Save strategy:", training_args.save_strategy)
    print("Save steps:", training_args.save_steps)
    print("Save total limit:", training_args.save_total_limit)
    print("Save safetensors:", training_args.save_safetensors)
    print("Resume from checkpoint:", training_args.resume_from_checkpoint)
    print("================================\n")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=True
    )

    # ------------------------
    # 2. Load dataset
    # ------------------------
    # "default" config contains all languages as per dataset card
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
    )

    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True,
        num_proc=32,
    )

    # ------------------------
    # 3. Train model
    # ------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args)
    )

    print("Detected last checkpoint:",
      get_last_checkpoint(training_args.output_dir))
    print("Resume arg:", training_args.resume_from_checkpoint)

    # Start training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save results
    trainer.save_model(os.path.join(store_base_dir, training_args.output_dir))

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args)
