#!/usr/bin/env python3
"""
Plot multiple slds heatmaps into a single grid image. Two layouts are supported:
default 2 rows × 3 cols, and an alternative 3 rows × 2 cols (variants × models).

Usage:
    python plot_slds_heatmap_grid.py [--bertscore] [--layout {grid,table}] [-o OUTPUT]

By default, the script scans results/*.json (top-level only), plots each file's
decision-vs-headnote matrix, and saves to results/plots/<metric>/all_<metric>_<layout>.pdf.

Flags:
    --judge (default): plot slds_judge
    --bertscore: plot BERTScore-F
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_slds_heatmap import load_matrix

TIMESLIKE_SERIF = [
    "Nimbus Roman",
    "TeX Gyre Termes",
    "Times New Roman",
    "Times",
    "DejaVu Serif",
    "serif",
]

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
        description="Generate slds heatmap grids from results JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to save the grid (default: results/plots/<metric>/all_<metric>_<layout>.pdf).",
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
        "--layout",
        choices=["grid", "table"],
        default="grid",
        help="Layout: 'grid' = 2 rows x 3 cols (models as rows), "
        "'table' = 3 rows x 2 cols (variants x models). Default: grid.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing results JSON files (top-level only; default: results/).",
    )
    return parser.parse_args()


LAYOUT_CONFIG = {
    # rows: 8B, 70B; cols: base, lora, full
    "grid": {
        "n_rows": 2,
        "n_cols": 3,
        "order": [
            "apertus8b_base",
            "apertus8b_lora",
            "apertus8b_full",
            "apertus70b_base",
            "apertus70b_lora",
            "apertus70b_full",
        ],
        "row_headers": ["Apertus 8B", "Apertus 70B"],
        "col_headers": ["zero-shot", "fine-tuned (LoRA)", "fine-tuned (full)"],
    },
    # rows: base/lora/full; cols: 8B, 70B
    "table": {
        "n_rows": 3,
        "n_cols": 2,
        "order": [
            "apertus8b_base",
            "apertus70b_base",
            "apertus8b_lora",
            "apertus70b_lora",
            "apertus8b_full",
            "apertus70b_full",
        ],
        "row_headers": ["zero-shot", "fine-tuned (LoRA)", "fine-tuned (full)"],
        "col_headers": ["Apertus 8B", "Apertus 70B"],
    },
}

# Human-friendly labels per run
STEM_LABELS = {
    "apertus8b_base": ("Apertus 8B", "zero-shot"),
    "apertus8b_lora": ("Apertus 8B", "fine-tuned (LoRA)"),
    "apertus8b_full": ("Apertus 8B", "fine-tuned (full)"),
    "apertus70b_base": ("Apertus 70B", "zero-shot"),
    "apertus70b_lora": ("Apertus 70B", "fine-tuned (LoRA)"),
    "apertus70b_full": ("Apertus 70B", "fine-tuned (full)"),
}


def find_json_files(json_dir: Path) -> Dict[str, Path]:
    paths_by_stem = {
        path.stem: path for path in json_dir.glob("*.json") if "deprecated" not in path.parts
    }
    return paths_by_stem


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


def plot_grid(
    matrices,
    vmin,
    vmax,
    metric: str,
    layout: str,
):
    sns.set_theme(
        style="white",
        rc={
            "font.family": "serif",
            "font.serif": TIMESLIKE_SERIF,
        },
    )
    config = LAYOUT_CONFIG[layout]
    n_rows = config["n_rows"]
    n_cols = config["n_cols"]
    # Size the figure by grid shape to avoid excessive whitespace (especially 3x2 layouts).
    cell_size_in = 2.55
    left_pad_in = 0.9 if layout == "table" else 0.6
    top_pad_in = 0.55
    bottom_pad_in = 0.45
    width_in = left_pad_in + (n_cols * cell_size_in)
    height_in = top_pad_in + bottom_pad_in + (n_rows * cell_size_in)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width_in, height_in),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.10, "hspace": 0.12},
    )
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
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
            annot_kws={"size": 11},
            square=True,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=11, pad=2)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title("")  # no per-panel titles

    # turn off unused axes
    for ax in list(axes_flat)[len(matrices) :]:
        ax.axis("off")

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.06, top=0.90)
    return fig, axes


def main():
    args = parse_args()
    metric = "BERTScore-F" if args.bertscore else "slds_judge"

    paths_by_stem = find_json_files(args.json_dir)
    if not paths_by_stem:
        raise SystemExit(f"No JSON files found in {args.json_dir}")

    matrices, vmin, vmax = collect_matrices(list(paths_by_stem.values()), metric=metric)

    if args.output:
        output_path = args.output
    else:
        plots_root = Path(__file__).parent / "plots"
        metric_dir = "bertscore" if metric.lower().startswith("bert") else "judge"
        output_path = plots_root / metric_dir / f"all_{metric_dir}_{args.layout}.pdf"
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".pdf")

    # Reorder matrices according to layout configuration
    order = LAYOUT_CONFIG[args.layout]["order"]
    paths_by_stem: Dict[str, Tuple[Path, np.ndarray, list, list]] = {
        p.stem: (p, m, d, h) for p, m, d, h in matrices
    }
    missing = [stem for stem in order if stem not in paths_by_stem]
    if missing:
        raise SystemExit(f"Missing expected JSON files: {', '.join(missing)}")
    ordered_matrices = [paths_by_stem[stem] for stem in order]

    # Add shared headers (row + column) instead of per-panel titles.
    def add_shared_headers(fig, axes):
        row_labels = LAYOUT_CONFIG[args.layout]["row_headers"]
        col_labels = LAYOUT_CONFIG[args.layout]["col_headers"]
        # Column headers (top)
        top_axes = axes[0]
        for idx, title in enumerate(col_labels):
            pos = top_axes[idx].get_position()
            x = pos.x0 + pos.width / 2
            y = pos.y1 + 0.010
            fig.text(x, y, title, ha="center", va="bottom", fontsize=13)
        # Row headers (left)
        row_label_offset = 0.075 if args.layout == "table" else 0.060
        for idx, title in enumerate(row_labels):
            pos = axes[idx][0].get_position()
            x = max(pos.x0 - row_label_offset, 0.01)
            y = pos.y0 + pos.height / 2
            fig.text(x, y, title, ha="right", va="center", rotation=90, fontsize=13)

    fig, axes = plot_grid(ordered_matrices, vmin, vmax, metric, layout=args.layout)

    axes_matrix = np.array(axes).reshape(
        LAYOUT_CONFIG[args.layout]["n_rows"],
        LAYOUT_CONFIG[args.layout]["n_cols"],
    )
    add_shared_headers(fig, axes_matrix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved grid heatmap to {output_path}")


if __name__ == "__main__":
    main()
