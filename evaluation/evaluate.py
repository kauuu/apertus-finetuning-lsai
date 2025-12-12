import os
from contextlib import suppress
from pathlib import Path
from textwrap import dedent

import typer
from dotenv import load_dotenv
import nltk
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

load_dotenv()

app = typer.Typer(pretty_exceptions_enable=False)


SLDS_GENERATION_SYSTEM_PROMPT = dedent(
    """
    You are a legal expert specializing in Swiss Federal Supreme Court decisions with extensive knowledge of legal terminology and conventions in German, French, and Italian. Your task is to generate a headnote for a provided leading decision. A headnote is a concise summary that captures the key legal points and significance of the decision. It is not merely a summary of the content but highlights the aspects that make the decision "leading" and important for future legislation.

    When generating the headnote:

    1. Focus on the core legal reasoning and key considerations that establish the decision's significance.
    2. Include any relevant references to legal articles (prefixed with "Art.") and considerations (prefixed with "E." in German or "consid." in French/Italian).
    3. Use precise legal terminology and adhere to the formal and professional style typical of Swiss Federal Supreme Court headnotes.
    4. Ensure clarity and coherence, so the headnote is logically structured and easy to understand in the specified language.

    Your response should consist solely of the headnote in the language specified by the user prompt.
    """
)


@app.command()
def main(
    model: str = typer.Option(default="/users/mneuwinger/scratch/apertus-finetuning-lsai/evaluation/merged_model", help="Model name or path"),
    one_shot: bool = typer.Option(default=True, help="Use one-shot prompting, true by default, set to false for trained models."),
    decision_language: str = typer.Option(default="de,fr,it", help="Language of the decision to summarize (de, fr, it), can be comma-separated for multiple languages. Default is de,fr,it"),
    headnote_language: str = typer.Option(default="de,fr,it", help="Language of the headnote to generate (de, fr, it), can be comma-separated for multiple languages. Default is de,fr,it"),
    debug: bool = typer.Option(default=False, help="Debug mode, uses only one sample per language pair"),
    tensor_parallel_size: int = typer.Option(default=1, help="Number of GPUs to shard the model across via tensor parallelism."),
):
    if tensor_parallel_size < 1:
        raise typer.BadParameter("tensor_parallel_size must be >= 1")

    known_languages = {"de", "fr", "it"}
    decision_languages = [lang.strip() for lang in decision_language.split(",")]
    headnote_languages = [lang.strip() for lang in headnote_language.split(",")]

    if not all(lang in known_languages for lang in decision_languages):
        raise ValueError(f"Unknown decision language(s) provided. Known languages are: {', '.join(known_languages)}")

    if not all(lang in known_languages for lang in headnote_languages):
        raise ValueError(f"Unknown headnote language(s) provided. Known languages are: {', '.join(known_languages)}")

    # Pre-warm and silence NLTK downloads (e.g., punkt_tab) to avoid noisy repeated stdout during metrics.
    nltk_data_dir = Path.home() / "nltk_data"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NLTK_DATA", str(nltk_data_dir))

    with suppress(Exception):
        nltk.download("punkt_tab", quiet=True)

    _orig_nltk_download = nltk.download

    def _quiet_nltk_download(*args, **kwargs):
        kwargs.setdefault("quiet", True)
        kwargs.setdefault("raise_on_error", False)
        try:
            return _orig_nltk_download(*args, **kwargs)
        except Exception:
            return None

    nltk.download = _quiet_nltk_download

    if debug:
        os.environ["SLDS_DEBUG_JUDGE"] = "1"

    evaluation_tracker = EvaluationTracker(
        output_dir="./results/",
    )

    task_file = Path(__file__).parent / "custom_task" / "slds.py"

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        custom_tasks_directory=str(task_file.resolve()),
        load_tasks_multilingual=True,
        max_samples=1 if debug else None,
    )

    generation_params = {
        "seed": 2025,
        "repetition_penalty": 1.05,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
    }

    if any([s in model for s in ["openai", "gpt", "deepseek"]]):
        generation_params.pop("repetition_penalty")
        generation_params.pop("top_k")

    if "anthropic" in model:
        generation_params.pop("repetition_penalty")

    generation_config = GenerationParameters(**generation_params)

    model_config = VLLMModelConfig(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_length=8192,
        swap_space=8,
        max_num_seqs=64,
        max_num_batched_tokens=2048,
        seed=2025,
        trust_remote_code=True,
        add_special_tokens=True,
        enable_prefix_caching=False,
        generation_parameters=generation_config,
        system_prompt=SLDS_GENERATION_SYSTEM_PROMPT.strip(),
    )

    tasks = [f"community|slds:{lang1}_{lang2}|{int(one_shot)}" for lang1 in decision_languages for lang2 in headnote_languages]
    tasks = ",".join(tasks)

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()

    pipeline.show_results()
    results = pipeline.get_results()
    pipeline.save_and_push_results()


if __name__ == "__main__":
    app()
