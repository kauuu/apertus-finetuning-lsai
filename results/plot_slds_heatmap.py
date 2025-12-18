#!/usr/bin/env python3
"""
Plot an slds_judge heatmap from a lighteval results JSON.

Usage:
    python plot_slds_heatmap.py results/results/8b_raw?.json

The script reads the requested JSON file, extracts `slds_judge` scores for
language pairs (decision_language -> headnote_language) and renders a 2D
heatmap. Use --title to override the plot title and -o to choose an output
path. By default the plot is saved to results/plots/<metric>/<json_stem>_<metric>_heatmap.png.

Flags:
    --judge (default): plot slds_judge
    --bertscore: plot BERTScore-F
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


TIMESLIKE_SERIF = [
    "Nimbus Roman",
    "TeX Gyre Termes",
    "Times New Roman",
    "Times",
    "DejaVu Serif",
    "serif",
]

# Prefer a Times-like serif font
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": TIMESLIKE_SERIF,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a heatmap of slds_judge scores for language pairs."
    )
    parser.add_argument("json_path", type=Path, help="Path to lighteval results JSON.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to save the plot (default: results/plots/<metric>/<json_stem>_<metric>_heatmap.pdf).",
    )
    score_group = parser.add_mutually_exclusive_group()
    score_group.add_argument(
        "--judge",
        action="store_true",
        help="Plot judge score (slds_judge). Default if no flag is given.",
    )
    score_group.add_argument(
        "--bertscore",
        action="store_true",
        help="Plot BERTScore-F instead of judge.",
    )
    parser.add_argument(
        "--title",
        help="Optional plot title (default: input filename stem).",
    )
    return parser.parse_args()


def preferred_lang_order(langs: Set[str]) -> List[str]:
    """Return a stable language order, preferring de/fr/it when present."""
    base_order = ["de", "fr", "it"]
    ordered = [lang for lang in base_order if lang in langs]
    extras = sorted(lang for lang in langs if lang not in base_order)
    return ordered + extras


def load_matrix(
    data: Dict, metric: str = "slds_judge"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a matrix of metric values indexed by (decision_lang, headnote_lang)."""
    results = data.get("results", {})
    lang_pairs = []
    for key in results:
        if not key.startswith("slds:") or "_average" in key or key == "all":
            continue
        pair = key.split("slds:", 1)[1].split("|", 1)[0]
        if "_" not in pair:
            continue
        decision_lang, headnote_lang = pair.split("_", 1)
        lang_pairs.append((decision_lang, headnote_lang))

    decision_langs = preferred_lang_order({pair[0] for pair in lang_pairs})
    headnote_langs = preferred_lang_order({pair[1] for pair in lang_pairs})

    matrix = np.full((len(decision_langs), len(headnote_langs)), np.nan)
    for key, values in results.items():
        if not key.startswith("slds:") or "_average" in key or key == "all":
            continue
        pair = key.split("slds:", 1)[1].split("|", 1)[0]
        if "_" not in pair:
            continue
        decision_lang, headnote_lang = pair.split("_", 1)
        if decision_lang in decision_langs and headnote_lang in headnote_langs:
            row = decision_langs.index(decision_lang)
            col = headnote_langs.index(headnote_lang)
            matrix[row, col] = values.get(metric, np.nan)

    return matrix, decision_langs, headnote_langs


def global_range(json_dir: Path, metric: str) -> Tuple[float, float]:
    """Compute a global vmin/vmax across all top-level (non-deprecated) JSONs."""
    vmin = float("inf")
    vmax = float("-inf")
    for path in json_dir.glob("*.json"):
        if "deprecated" in path.parts:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            matrix, _, _ = load_matrix(data, metric=metric)
            if not np.isnan(matrix).all():
                vmin = min(vmin, np.nanmin(matrix))
                vmax = max(vmax, np.nanmax(matrix))
        except Exception:
            continue
    if vmin == float("inf") or vmax == float("-inf"):
        return np.nan, np.nan
    return vmin, vmax


def plot_heatmap(
    matrix: np.ndarray,
    decision_langs: List[str],
    headnote_langs: List[str],
    metric: str,
    output_path: Path,
    vmin: float,
    vmax: float,
):
    sns.set_theme(
        style="white",
        rc={
            "font.family": "serif",
            "font.serif": TIMESLIKE_SERIF,
        },
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # Fallback to local range if global was unavailable
    if np.isnan(vmin) or np.isnan(vmax):
        vmin = np.nanmin(matrix)
        vmax = np.nanmax(matrix)
    cmap = "Blues" if metric.lower().startswith("bert") else "Reds"
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=headnote_langs,
        yticklabels=decision_langs,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": metric},
        square=True,
        annot_kws={"size": 12},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.tick_params(axis="both", which="major", labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved heatmap to {output_path}")


def main():
    args = parse_args()
    with args.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if args.bertscore:
        metric = "BERTScore-F"
    else:
        metric = "slds_judge"

    matrix, decision_langs, headnote_langs = load_matrix(data, metric=metric)

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        plots_root = args.json_path.parent.parent / "plots"
        metric_dir = "bertscore" if metric.lower().startswith("bert") else "judge"
        plots_dir = plots_root / metric_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        sanitized_metric = metric.replace("/", "-")
        output_path = plots_dir / f"{args.json_path.stem}_{sanitized_metric}_heatmap.pdf"
    # Enforce PDF extension unless user explicitly provided another one.
    if args.output is None and output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")

    # Compute a global color range across top-level JSONs in the same directory.
    g_vmin, g_vmax = global_range(args.json_path.parent, metric)

    plot_heatmap(
        matrix,
        decision_langs,
        headnote_langs,
        metric,
        output_path,
        g_vmin,
        g_vmax,
    )


if __name__ == "__main__":
    main()
