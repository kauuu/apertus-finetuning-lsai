# Evaluation with the LoRA finetuned model

## Setting up the Environment

You need to build the Dockerfile to get the image to run vllm!

To build the image, run `podman build -t vllm:1` on a compute node and then save the image as a `sqsh` file. Once you have the sqsh file, you can use it as the image on the `apertus_evaluation.toml` file. 

Run the following command to get a compute node with the environment to run VLLM: 

```zsh
srun -A large-sc-2 --environment=<path-to-toml-file> -p debug --pty bash
```

## Running the LoRa finetuned model

Once you are on the compute node with the right environment, run the following code to code to start the server with the LoRa finetuned model (if you have it stored locally already):

```zsh
vllm serve swiss-ai/Apertus-8B-Instruct-2509 --enable-lora --lora-modules "my_finetune=/users/$USER/scratch/apertus-finetuning-lsai/finetuning/Apertus-FT/output/apertus_lora_r64_ctx8k/checkpoint-1500" --max-lora-rank 32
```

Alternative command if some additional parameters: 

```zsh
vllm serve swiss-ai/Apertus-8B-Instruct-2509 --enable-lora --lora-modules "my_finetune=/users/$USER/scratch/apertus-finetuning-lsai/finetuning/Apertus-FT/output/apertus_lora_r64_ctx8k/checkpoint-1500" --max-model-len 32000 --enable-prefix-caching --seed 2025 --gpu-memory-utilization 0.7 --max-lora-rank 32
```

If you do not have the model stored locally, then there is a finetuned version of the model hosted on huggingface, and it can be run by: 

```zsh
vllm serve kkaushik02/apertus_finetuned_merged_model_r32  --served-model-name "my_finetune" --max-model-len 32000     --gpu-memory-utilization 0.9 --dtype float32 > vllm.log 2>&1 &
```

Make sure to have `--dtype float32`!

Moreover, if you want to run the hosted model in the background, without it taking up your terminal, end the above command with ` > vllm.log 2>&1 &`. This ensures you get back control of the terminal, and the logs are saved to `vllm.log` file. You can check the logs using `tail -f vllm.log`.

## Running the Evaluation Script

1. Run `uv sync`
2. Run `unset SSL_CERT_FILE`
3. Run `uv pip install language_data`
4. run the uv run eval command: `uv run evaluate.py --no-one-shot --model hosted_vllm/my_finetune`

To avoid disk quota issues with Huggingface cache, set the following environment variables in your shell before running the evaluation:

```zsh
export HF_HOME="/iopsstor/scratch/cscs/kkarthikeyan/.cache/huggingface"
export HF_DATASETS_CACHE="/iopsstor/scratch/cscs/kkarthikeyan/.cache/huggingface/datasets"
export UV_CACHE_DIR="/users/kkarthikeyan/scratch/.cache/uv"
```

# Problems

I have the `/xielu` cuda activation code, however running apertus model after setting it up leads to failures with the logger. However, without XIELU, inference is really slow, needs to be fixed!
