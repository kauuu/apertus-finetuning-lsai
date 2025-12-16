#!/usr/bin/env python3
"""
Plot multiple slds heatmaps into a single 3x3 grid image.

Usage:
    python plot_slds_heatmap_grid.py [--bertscore] [-o OUTPUT]

By default, the script scans results/results/*.json, plots each file's
decision-vs-headnote matrix, and saves to
results/plots/<metric>/all_<metric>_grid.png.

Flags:
    --judge (default): plot slds_judge
    --bertscore: plot BERTScore-F
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_slds_heatmap import load_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3x3 grid of slds heatmaps from results JSON files."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to save the grid (default: results/plots/<metric>/all_<metric>_grid.png).",
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
        "--json-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing results JSON files (default: results/results).",
    )
    return parser.parse_args()


def find_json_files(json_dir: Path) -> List[Path]:
    return sorted(json_dir.glob("*.json"))


def collect_matrices(json_paths: List[Path], metric: str) -> Tuple[list, float, float]:
    matrices = []
    vmin = float("inf")
    vmax = float("-inf")
    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        matrix, decision_langs, headnote_langs = load_matrix(data, metric=metric)
        matrices.append((path, matrix, decision_langs, headnote_langs))
        if not np.isnan(matrix).all():
            vmin = min(vmin, np.nanmin(matrix))
            vmax = max(vmax, np.nanmax(matrix))
    if vmin == float("inf"):
        vmin = np.nan
    if vmax == float("-inf"):
        vmax = np.nan
    return matrices, vmin, vmax


def plot_grid(matrices, vmin, vmax, metric: str, output_path: Path):
    sns.set(style="white")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes_flat = axes.flat
    cmap = "Blues" if metric.lower().startswith("bert") else "Reds"

    for ax, item in zip(axes_flat, matrices):
        path, matrix, decision_langs, headnote_langs = item
        if np.isnan(matrix).all():
            ax.axis("off")
            continue
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            xticklabels=headnote_langs,
            yticklabels=decision_langs,
            vmin=vmin if np.isfinite(vmin) else None,
            vmax=vmax if np.isfinite(vmax) else None,
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Headnote Lang")
        ax.set_ylabel("Decision Lang")
        ax.set_title(path.stem)

    # turn off unused axes
    for ax in list(axes_flat)[len(matrices) :]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved grid heatmap to {output_path}")


def main():
    args = parse_args()
    metric = "BERTScore-F" if args.bertscore else "slds_judge"

    json_paths = find_json_files(args.json_dir)
    if not json_paths:
        raise SystemExit(f"No JSON files found in {args.json_dir}")

    matrices, vmin, vmax = collect_matrices(json_paths, metric=metric)

    if args.output:
        output_path = args.output
    else:
        plots_root = Path(__file__).parent / "plots"
        metric_dir = "bertscore" if metric.lower().startswith("bert") else "judge"
        output_path = plots_root / metric_dir / f"all_{metric_dir}_grid.png"

    plot_grid(matrices, vmin, vmax, metric, output_path)


if __name__ == "__main__":
    main()
