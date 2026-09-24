#!/bin/bash
set -euo pipefail

MODE=${1:-energy}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
FT_MASTER_PORT=${FT_MASTER_PORT:-6672}
INFER_MASTER_PORT=${INFER_MASTER_PORT:-6673}
LLAMA_CKPT_PATH=${LLAMA_CKPT_PATH:-/nfs1/outdated/WYT/models/Llama-2-7b-chat-hf}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-6}
GRADIENT_CALIBRATION_SAMPLES=${GRADIENT_CALIBRATION_SAMPLES:-8}
GRADIENT_CALIBRATION_BATCH_SIZE=${GRADIENT_CALIBRATION_BATCH_SIZE:-1}
DASH_ABLATION_SEED=${DASH_ABLATION_SEED:-123}

export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=${WANDB_PROJECT:-finetune-llama-avqa}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_NAME=llama2-7b-avqa-dash-ablation-${MODE}-ratio3
RUN_DIR=results/${WANDB_PROJECT}/${RUN_NAME}
TOPOLOGY_PATH=${RUN_DIR}/dash_ablation_topology.json
LOG_SUFFIX=${DASH_ABLATION_LOG_SUFFIX:-}
if [[ -n "$LOG_SUFFIX" ]]; then
    FT_LOG=results/log/llama2/avqa/dash-ablation-${MODE}-ratio3-${LOG_SUFFIX}-ft.log
else
    FT_LOG=results/log/llama2/avqa/dash-ablation-${MODE}-ratio3-ft.log
fi
mkdir -p "$(dirname "$FT_LOG")" "$RUN_DIR"
: > "$FT_LOG"

torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port "$FT_MASTER_PORT" \
    scripts/finetune/finetune_dash_ablation.py \
    --dash_ablation_mode "$MODE" \
    --dash_ablation_seed "$DASH_ABLATION_SEED" \
    --dash_ablation_topology_path "$TOPOLOGY_PATH" \
    --dash_gradient_calibration_samples "$GRADIENT_CALIBRATION_SAMPLES" \
    --dash_gradient_calibration_batch_size "$GRADIENT_CALIBRATION_BATCH_SIZE" \
    --deepspeed deepspeed/stage1-offload.json \
    --llm_name llama --model_name_or_path "$LLAMA_CKPT_PATH" \
    --loramethod train --reserved_modality None --exp_desc "dash-ablation-${MODE}-ratio3" \
    --dash_lora_safe_importance True \
    --freeze_backbone True --lora_enable True --bits 32 \
    --lora_r 4 --lora_alpha 16 --lora_dropout 0.05 \
    --top_k_layers 65 --ratio 3 --blc_weight 1 --blc_alpha 1 \
    --bf16 False --tf32 True --fp16 False \
    --avqa_task True --ave_task False \
    --save_modules vl_projector,al_projector,lora \
    --visual_branch True --video_frame_nums 10 \
    --vit_ckpt_path /nfs1/outdated/WYT/models/clip-vit-large-patch14 \
    --select_feature patch --image_size 224 --patch_size 14 \
    --visual_query_token_nums 32 --audio_branch True \
    --BEATs_ckpt_path /nfs1/outdated/WYT/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir "$RUN_DIR" \
    --num_train_epochs 3 --per_device_train_batch_size "$LOCAL_BATCH_SIZE" \
    --per_device_eval_batch_size "$LOCAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --ddp_find_unused_parameters True --evaluation_strategy no \
    --save_strategy steps --save_steps 0.1 --save_total_limit 10 \
    --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.03 \
    --lr_scheduler_type cosine --logging_steps 1 \
    --gradient_checkpointing True --half_precision_backend auto \
    --dataloader_num_workers 4 --report_to tensorboard >> "$FT_LOG" 2>&1

if [[ "${RUN_INFER:-1}" == "0" ]]; then
    echo "Finished FT only for $MODE"
    echo "Run dir: $RUN_DIR"
    echo "Topology: $TOPOLOGY_PATH"
    echo "FT log: $FT_LOG"
    exit 0
fi

CKPT_DIR=$(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
if [[ -z "$CKPT_DIR" ]]; then
    echo "No checkpoint-* directory found under $RUN_DIR" >&2
    exit 1
fi

INFER_LOG=${CKPT_DIR}/inference_avqa_dash_ablation_${MODE}.log

torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port "$INFER_MASTER_PORT" \
    scripts/infer/inference_cut_multiGPU_dash_ablation.py \
    --seed 42 --mode test \
    --dash_ablation_mode "$MODE" \
    --dash_ablation_seed "$DASH_ABLATION_SEED" \
    --dash_ablation_topology_path "$TOPOLOGY_PATH" \
    --llm_name llama --model_name_or_path "$LLAMA_CKPT_PATH" \
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
    --audio_query_token_nums 32 --output_dir not_used >> "$INFER_LOG" 2>&1

echo "Finished $MODE"
echo "Run dir: $RUN_DIR"
echo "Checkpoint: $CKPT_DIR"
echo "Topology: $TOPOLOGY_PATH"
echo "FT log: $FT_LOG"
echo "Infer log: $INFER_LOG"
