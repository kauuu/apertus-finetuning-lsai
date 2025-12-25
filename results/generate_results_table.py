#!/usr/bin/env python3
"""
Generate a LaTeX table with aggregated ("all") metrics for each results JSON.

By default, the script scans results/*.json (top-level only, skipping deprecated),
pulls the "all" metrics, and writes a booktabs-style LaTeX table to results/plots/all_results_table.tex
while also printing it to stdout. Use --output to change the destination.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


# Human-friendly model labels for each run (keyed by JSON stem)
MODEL_LABELS: Dict[str, str] = {
    # Apertus
    "apertus8b_base": "Apertus 8B",
    "apertus8b_base_oneshot": "Apertus 8B",
    "apertus8b_full": "Apertus 8B",
    "apertus8b_lora": "Apertus 8B",
    "apertus70b_base": "Apertus 70B",
    "apertus70b_base_oneshot": "Apertus 70B",
    "apertus70b_full": "Apertus 70B",
    "apertus70b_lora": "Apertus 70B",
    # Non-Apertus
    "phimini": "Phi-3.5-mini",
    "llama3b": "Llama 3.2 3B",
    "qwen0.5b": "Qwen2.5 0.5B",
    "qwen1.5b": "Qwen2.5 1.5B",
    "qwen3b": "Qwen2.5 3B",
    "qwen7b": "Qwen2.5 7B",
    "qwen14b": "Qwen2.5 14B",
}

# Preferred ordering of rows; anything unlisted is appended alphabetically
RUN_ORDER = [
    "phimini",
    "llama3b",
    "qwen0.5b",
    "qwen1.5b",
    "qwen3b",
    "qwen7b",
    "qwen14b",
    "apertus8b_base",
    "apertus8b_base_oneshot",
    "apertus8b_lora",
    "apertus8b_full",
    "apertus70b_base",
    "apertus70b_base_oneshot",
    "apertus70b_lora",
    "apertus70b_full",
]

METRICS = [
    ("BERTScore-F", "BERTScore $\\uparrow$"),
    ("bleu", "BLEU $\\uparrow$"),
    ("rouge1", "ROUGE-1 $\\uparrow$"),
    ("rouge2", "ROUGE-2 $\\uparrow$"),
    ("rougeL", "ROUGE-L $\\uparrow$"),
    ("slds_judge", "JUDGE $\\uparrow$"),
]

SCALE_FACTORS = {
    # Show ROUGE as percentages instead of fractions
    "rouge1": 100.0,
    "rouge2": 100.0,
    "rougeL": 100.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LaTeX table from aggregated (all) metrics."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing results JSON files (top-level only; default: results/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "plots" / "all_results_table.tex",
        help="Path to write the LaTeX table (default: results/plots/all_results_table.tex).",
    )
    return parser.parse_args()


def find_json_files(json_dir: Path) -> List[Path]:
    return sorted(
        p for p in json_dir.glob("*.json") if "deprecated" not in p.parts
    )


def format_metric(entry: Dict, key: str, scale: float = 1.0) -> str:
    mean = entry.get(key)
    stderr = entry.get(f"{key}_stderr")
    if mean is None:
        return "-"

    scaled_mean = mean * scale
    if stderr is None:
        return f"{scaled_mean:.2f}"

    scaled_stderr = stderr * scale
    return f"{scaled_mean:.2f} ± {scaled_stderr:.2f}"


def variant_from_stem(stem: str) -> str:
    lowered = stem.lower()
    if "oneshot" in lowered:
        return "one-shot"
    if "full" in lowered:
        return "Full fine-tuned"
    if "lora" in lowered:
        return "LoRA fine-tuned"
    if "base" in lowered:
        return "zero-shot"
    return "LoRA fine-tuned"


def model_label_from_stem(stem: str, data: Dict) -> str:
    if stem in MODEL_LABELS:
        return MODEL_LABELS[stem]
    # Gracefully fall back to a label without a trailing "_oneshot" if present.
    if stem.endswith("_oneshot") and stem.removesuffix("_oneshot") in MODEL_LABELS:
        return MODEL_LABELS[stem.removesuffix("_oneshot")]
    return data.get("config_general", {}).get("model_name", stem)


def build_rows(json_paths: List[Path]) -> List[Dict]:
    rows = []
    for path in json_paths:
        if "deprecated" in path.parts:
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        all_metrics = data.get("results", {}).get("all")
        if not all_metrics:
            # Skip files without the aggregated block
            continue
        model = model_label_from_stem(path.stem, data)
        row = {
            "stem": path.stem,
            "model": model,
            "setting": variant_from_stem(path.stem),
        }
        for key, _ in METRICS:
            row[key] = format_metric(
                all_metrics,
                key,
                scale=SCALE_FACTORS.get(key, 1.0),
            )
        rows.append(row)
    return rows


def order_rows(rows: List[Dict]) -> List[Dict]:
    order_map = {name: idx for idx, name in enumerate(RUN_ORDER)}
    return sorted(
        rows,
        key=lambda r: (order_map.get(r["stem"], len(RUN_ORDER)), r["stem"]),
    )


def to_latex(rows: List[Dict]) -> str:
    header_cols = ["Model", "Variant"] + [label for _, label in METRICS]
    col_spec = "ll" + "c" * len(METRICS)

    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule"]
    lines.append(" & ".join(header_cols) + " \\\\")
    lines.append("\\midrule")

    for row in rows:
        values = [row["model"], row["setting"]]
        values.extend(row[key] for key, _ in METRICS)
        lines.append(" & ".join(values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main():
    args = parse_args()
    json_paths = find_json_files(args.json_dir)
    if not json_paths:
        raise SystemExit(f"No JSON files found in {args.json_dir}")

    rows = order_rows(build_rows(json_paths))
    if not rows:
        raise SystemExit("No rows generated (missing 'all' metrics?).")

    table_tex = to_latex(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table_tex, encoding="utf-8")
    print(table_tex)
    print(f"\nSaved LaTeX table to {args.output}")


if __name__ == "__main__":
    main()
