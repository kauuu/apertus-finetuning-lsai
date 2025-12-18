# Evaluation quick start

## 1) Build the container and create a squashfs
- Use the provided `Dockerfile` with `srun` to build the image on the cluster node.
- Convert the built image to a `.sqsh` (squashfs) file.
- Update the path to that `.sqsh` in `evaluation.toml` so jobs mount the right container.

## 2) Install the evaluation environment
- Submit `setup_and_run.sbatch` to create/refresh the venv (`venv-eval`) inside the container.
- Logs land in `logs/setup_*.out|err`.

## 3) Choose your generator model
- Default full run (`run_evaluation.sbatch`) uses the base Apertus model: `swiss-ai/Apertus-8B-Instruct-2509` (overridable via `MODEL_PATH`).
- Debug run (`run_evaluation_debug.sbatch`) defaults to the merged LoRA model at `evaluation/merged_model` (overridable via `MODEL_PATH`).
- To test another model, pass `--export=MODEL_PATH=<hf_id_or_path>` when submitting the sbatch.

## 4) Judge model
- Judge runs as a separate vLLM server. Default: `JUDGE_MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Adjust by exporting `JUDGE_MODEL_ID` when submitting: `sbatch ... --export=MODEL_PATH=...,JUDGE_MODEL_ID=<hf_id>`.

## 5) Run evaluations
- Full test set (all languages): `sbatch run_evaluation.sbatch`.
- Debug one sample per lang (de→de): `sbatch run_evaluation_debug.sbatch`.
- Outputs: metrics in `results/` and logs in `logs/`.

## 6) Merging a LoRA adapter
- If you have a LoRA checkpoint, merge it into a standalone model with `merge_lora.sbatch`.
- Defaults: base `swiss-ai/Apertus-8B-Instruct-2509`, adapter path in finetuning output, merged model saved to `evaluation/merged_model`.
- Override via `BASE_MODEL`, `ADAPTER_PATH`, `OUTPUT_PATH` when submitting.

## 7) Notes
- `evaluation.toml` is used by `srun --environment=...` in the sbatch scripts; keep it in sync with the squashfs path.
- Logs for judge server and generator appear in `logs/eval_*.out|err` (or `_debug_`).
- `run_evaluation_openrouter.sbatch` runs from a per-job node-local working dir to avoid LiteLLM/diskcache SQLite errors on shared filesystems (`sqlite3.OperationalError: locking protocol`) and writes artifacts to `results/<jobid>_<model>_<shotmode>/`.
