#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_dir="${script_dir}/results"

shopt -s nullglob
json_files=("${results_dir}"/*.json)
shopt -u nullglob

if [[ ${#json_files[@]} -eq 0 ]]; then
  echo "No JSON files found in ${results_dir}" >&2
  exit 1
fi

for json in "${json_files[@]}"; do
  echo "Plotting ${json}"
  "${PYTHON_BIN}" "${script_dir}/plot_slds_heatmap.py" "${json}"
  "${PYTHON_BIN}" "${script_dir}/plot_slds_heatmap.py" "${json}" --bertscore
done

echo "Plotting grid (judge)"
"${PYTHON_BIN}" "${script_dir}/plot_slds_heatmap_grid.py" --layout grid
"${PYTHON_BIN}" "${script_dir}/plot_slds_heatmap_grid.py" --layout table

echo "Plotting grid (bertscore)"
"${PYTHON_BIN}" "${script_dir}/plot_slds_heatmap_grid.py" --bertscore --layout grid
"${PYTHON_BIN}" "${script_dir}/plot_slds_heatmap_grid.py" --bertscore --layout table

echo "Generating LaTeX table"
"${PYTHON_BIN}" "${script_dir}/generate_results_table.py"
