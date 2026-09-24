#!/bin/bash

NPROC_PER_NODE=4
MASTER_PORT=6673
llama_ckpt_path=/nfs1/outdated/WYT/models/Llama-2-7b-chat-hf
RUN_DIR=results/finetune-llama-ave/llama2-7b-ave-dash-scored-ratio1
OUTPUT_LOG=results/log/llama2/infer-dash-scored-ratio1.log

export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
mkdir -p "$(dirname "$OUTPUT_LOG")"

# Override with CKPT_DIR=/path/to/checkpoint when evaluating a specific step.
if [ -z "$CKPT_DIR" ]; then
    CKPT_DIR=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
fi
if [ -z "$CKPT_DIR" ] || [ ! -f "$CKPT_DIR/finetune_weights.bin" ]; then
    echo "No completed checkpoint with finetune_weights.bin found under $RUN_DIR" >&2
    exit 1
fi

torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
    scripts/infer/inference_cut_multiGPU.py \
    --seed 42 --mode test --llm_name llama \
    --reserved_modality None --loramethod test --cut_folds 0 \
    --model_name_or_path "$llama_ckpt_path" --freeze_backbone True \
    --lora_enable True --bits 32 --lora_r 4 --lora_alpha 16 \
    --lora_dropout 0.05 --ratio 1 --top_k_layers 56 \
    --dash_lora_safe_importance True \
    --blc_weight 1 --blc_alpha 1 --bf16 False --tf32 False --fp16 False \
    --ckpt_dir "$CKPT_DIR" --avqa_task False --ave_task True \
    --device cuda:0 --visual_branch True --video_frame_nums 10 \
    --vit_ckpt_path /nfs1/outdated/WYT/models/clip-vit-large-patch14 \
    --image_size 224 --patch_size 14 --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /nfs1/outdated/WYT/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 --output_dir not_used >> "$OUTPUT_LOG" 2>&1
