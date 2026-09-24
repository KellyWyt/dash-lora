#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/nfs1/outdated/WYT/MokA-copy
PIPELINE_LOG="$PROJECT_DIR/results/log/llama2/until-noon-pipeline.log"
LAYERWISE_SESSION=llama2_ave_layerwise_countmatched
GPU_USE_SESSION=llama2-ave-until-noon

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$PIPELINE_LOG")"
exec >> "$PIPELINE_LOG" 2>&1

timestamp() {
    date '+%F %T %Z'
}

start_gpu_filler_until_noon() {
    local now noon remaining
    now=$(date +%s)
    noon=$(date -d 'today 12:00' +%s)
    remaining=$((noon - now))
    if (( remaining <= 0 )); then
        echo "[$(timestamp)] Noon reached; GPU filler not needed."
        return
    fi
    echo "[$(timestamp)] Starting GPU filler for ${remaining}s."
    GPU_USE_SESSION="$GPU_USE_SESSION" \
        bash gpu_use/start_gpu_use.sh "${remaining}s" 0,1,2,3 100 8192 23000
}

on_exit() {
    local status=$?
    if (( status != 0 )); then
        echo "[$(timestamp)] Pipeline failed with status $status; filling GPUs until noon."
        start_gpu_filler_until_noon || true
    fi
}
trap on_exit EXIT

echo "[$(timestamp)] Waiting for layerwise-shared FT."
while tmux has-session -t "$LAYERWISE_SESSION" 2>/dev/null; do
    sleep 30
done

echo "[$(timestamp)] Running layerwise-shared inference."
bash scripts/infer/infer_llama2-7b-ave-layerwise-shared-ratio1.sh

echo "[$(timestamp)] Running Dash-LoRA random-selection FT."
bash scripts/finetune/ft_llama2-7b-ave-dash-random-ratio1.sh

echo "[$(timestamp)] Running Dash-LoRA random-selection inference."
bash scripts/infer/infer_llama2-7b-ave-dash-random-ratio1.sh

echo "[$(timestamp)] All requested experiments completed."
start_gpu_filler_until_noon
trap - EXIT
