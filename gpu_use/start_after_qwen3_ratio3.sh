#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/nfs1/outdated/WYT/MokA-copy
PIPELINE_SESSION=qwen3_avqa_dash_r3
GPU_SESSION=qwen3-ratio3-gpu-use-24h
RESULT_FILE="$PROJECT_DIR/results/finetune-qwen3/qwen3-8B_avqa_dash-lora-r-4-alpha-16-ratio-3/checkpoint-2994/inference_avqa/bs1_bf16_mt500_test/merged_results.jsonl"
WATCH_LOG="$PROJECT_DIR/gpu_use/logs/qwen3-ratio3-watcher.log"

mkdir -p "$(dirname "$WATCH_LOG")"
echo "[$(date '+%F %T %Z')] Waiting for FT + infer session: $PIPELINE_SESSION" >> "$WATCH_LOG"

while tmux has-session -t "$PIPELINE_SESSION" 2>/dev/null; do
    sleep 15
done

if [[ ! -s "$RESULT_FILE" ]]; then
    echo "[$(date '+%F %T %Z')] Pipeline ended without merged inference output; GPU workload not started." >> "$WATCH_LOG"
    exit 1
fi

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    sleep 15
done

echo "[$(date '+%F %T %Z')] Inference complete and GPUs idle; starting 24h workload." >> "$WATCH_LOG"
cd "$PROJECT_DIR"
GPU_USE_SESSION="$GPU_SESSION" bash gpu_use/start_gpu_use.sh 24h 0,1,2,3 100 8192 23000 >> "$WATCH_LOG" 2>&1
