# How to get the sqsh file

You need to build the Dockerfile to get the image to run vllm!

To build the image, run `podman build -t vllm:1` on a compute node and then save the image as a `sqsh` file. Once you have the sqsh file, you can use it as the image on the `apertus_evaluation.toml` file. 

Run the following command to get a compute node with the environment to run VLLM: 

```zsh
srun -A large-sc-2 --environment=<path-to-toml-file> -p debug --pty bash
```

## Running the LoRa finetuned model

Once you are on the compute node with the right environment, run the following code to code to start the server with the LoRa finetuned model:

```zsh
vllm serve swiss-ai/Apertus-8B-Instruct-2509 --enable-lora --lora-modules "my_finetune=/users/$USER/scratch/apertus-finetuning-lsai/finetuning/Apertus-FT/output/apertus_lora_r64_ctx8k/checkpoint-1500"
```