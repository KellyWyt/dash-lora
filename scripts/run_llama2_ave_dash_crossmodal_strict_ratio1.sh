#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/nfs1/outdated/WYT/MokA-copy
LOG_FILE="$PROJECT_DIR/results/log/llama2/dash-crossmodal-strict-ratio1-pipeline.log"
RESULT_DIR="$PROJECT_DIR/results/finetune-llama-ave/llama2-7b-ave-dash-crossmodal-strict-ratio1"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
exec >> "$LOG_FILE" 2>&1


echo "[$(date '+%F %T %Z')] Starting strict cross-modal Dash-LoRA FT."
bash scripts/finetune/ft_llama2-7b-ave-dash-crossmodal-strict-ratio1.sh

echo "[$(date '+%F %T %Z')] Starting strict cross-modal Dash-LoRA inference."
bash scripts/infer/infer_llama2-7b-ave-dash-crossmodal-strict-ratio1.sh

INFER_DIR="$RESULT_DIR/checkpoint-207/inference_ave/bs1_tf32_mt150_seed42_test"
cp "$INFER_DIR/results0.jsonl" "$INFER_DIR/merged_results.jsonl"
sed -i '$r '"$INFER_DIR"'/results1.jsonl' "$INFER_DIR/merged_results.jsonl"
sed -i '$r '"$INFER_DIR"'/results2.jsonl' "$INFER_DIR/merged_results.jsonl"
sed -i '$r '"$INFER_DIR"'/results3.jsonl' "$INFER_DIR/merged_results.jsonl"

echo "[$(date '+%F %T %Z')] Merged inference results."
wc -l "$INFER_DIR/merged_results.jsonl"
