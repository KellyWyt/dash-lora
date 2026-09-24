#!/bin/bash
set -euo pipefail

PROJECT_DIR=/nfs1/outdated/WYT/MokA-copy
CKPT_DIR="$PROJECT_DIR/results/finetune-llama-avqa/llama2-7b-avqa-dash-ablation-global_energy-ratio3/checkpoint-1995"
TOPOLOGY_PATH="$PROJECT_DIR/results/finetune-llama-avqa/llama2-7b-avqa-dash-ablation-global_energy-ratio3/dash_ablation_topology.json"
LOG_FILE="$CKPT_DIR/inference_avqa_dash_ablation_global_energy.log"

cd "$PROJECT_DIR"
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

: > "$LOG_FILE"
/nfs1/miniconda3/envs/moka/bin/torchrun \
    --nproc_per_node 4 --master_port 6673 \
    scripts/infer/inference_cut_multiGPU_dash_ablation.py \
    --seed 42 --mode test \
    --dash_ablation_mode global_energy \
    --dash_ablation_seed 123 \
    --dash_ablation_topology_path "$TOPOLOGY_PATH" \
    --llm_name llama \
    --model_name_or_path /nfs1/outdated/WYT/models/Llama-2-7b-chat-hf \
    --reserved_modality None --loramethod test \
    --freeze_backbone True --lora_enable True --bits 32 \
    --lora_r 4 --lora_alpha 16 --lora_dropout 0.05 \
    --top_k_layers 65 --ratio 3 --dash_lora_safe_importance True \
    --blc_weight 1 --blc_alpha 1 \
    --bf16 False --tf32 True --fp16 False \
    --ckpt_dir "$CKPT_DIR" \
    --avqa_task True --ave_task False \
    --visual_branch True --video_frame_nums 10 \
    --vit_ckpt_path /nfs1/outdated/WYT/models/clip-vit-large-patch14 \
    --select_feature patch --image_size 224 --patch_size 14 \
    --visual_query_token_nums 32 --audio_branch True \
    --BEATs_ckpt_path /nfs1/outdated/WYT/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 --output_dir not_used >> "$LOG_FILE" 2>&1
