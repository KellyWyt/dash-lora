#!/bin/bash

NPROC_PER_NODE=4
MASTER_PORT=6672
llama_ckpt_path=/nfs1/outdated/WYT/models/Llama-2-7b-chat-hf
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=6

export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune-llama-ave
RUN_NAME=llama2-7b-ave-dash-random-ratio1
OUTPUT_LOG=results/log/llama2/dash-random-ratio1.log
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
mkdir -p "$(dirname "$OUTPUT_LOG")"

torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
    scripts/finetune/finetune_dash_random.py \
    --deepspeed deepspeed/stage1-offload.json \
    --llm_name llama --model_name_or_path "$llama_ckpt_path" \
    --loramethod train --loratype layerwise_shared \
    --dash_lora_safe_importance True \
    --reserved_modality None --exp_desc dash-lora-random-selection-ablation \
    --freeze_backbone True --lora_enable True --bits 32 \
    --lora_r 4 --lora_alpha 16 --lora_dropout 0.05 \
    --top_k_layers 56 --ratio 1 --blc_weight 1 --blc_alpha 1 \
    --bf16 False --tf32 True --fp16 False \
    --avqa_task False --ave_task True \
    --save_modules vl_projector,al_projector,lora \
    --visual_branch True --video_frame_nums 10 \
    --vit_ckpt_path /nfs1/outdated/WYT/models/clip-vit-large-patch14 \
    --select_feature patch --image_size 224 --patch_size 14 \
    --visual_query_token_nums 32 --audio_branch True \
    --BEATs_ckpt_path /nfs1/outdated/WYT/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir results/$WANDB_PROJECT/$RUN_NAME \
    --num_train_epochs 3 --per_device_train_batch_size "$LOCAL_BATCH_SIZE" \
    --per_device_eval_batch_size "$LOCAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --ddp_find_unused_parameters True --evaluation_strategy no \
    --save_strategy steps --save_steps 0.1 --save_total_limit 10 \
    --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.03 \
    --lr_scheduler_type cosine --logging_steps 1 \
    --gradient_checkpointing True --half_precision_backend auto \
    --dataloader_num_workers 4 --report_to tensorboard >> "$OUTPUT_LOG" 2>&1
