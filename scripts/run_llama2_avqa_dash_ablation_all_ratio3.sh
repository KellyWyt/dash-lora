#!/bin/bash
set -euo pipefail

MODES=(energy random_topology global_energy reverse_energy gradient)
for MODE in "${MODES[@]}"; do
    echo "===== $(date +%F_%T) START ${MODE} ====="
    bash scripts/run_llama2_avqa_dash_ablation_ratio3.sh "${MODE}"
    echo "===== $(date +%F_%T) END ${MODE} ====="
done
