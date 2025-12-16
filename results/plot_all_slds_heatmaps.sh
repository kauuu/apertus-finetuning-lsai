#!/usr/bin/env bash
set -euo pipefail

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
  python "${script_dir}/plot_slds_heatmap.py" "${json}"
  python "${script_dir}/plot_slds_heatmap.py" "${json}" --bertscore
done

echo "Plotting grid (judge)"
python "${script_dir}/plot_slds_heatmap_grid.py"

echo "Plotting grid (bertscore)"
python "${script_dir}/plot_slds_heatmap_grid.py" --bertscore
