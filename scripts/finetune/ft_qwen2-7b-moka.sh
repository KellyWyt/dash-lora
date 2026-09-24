#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=4
MASTER_PORT=6666
RANK=0

# qwen2_ckpt_path=pretrained_ckpts/Qwen2-1.5B-Instruct
qwen2_ckpt_path=pretrained_ckpts/Qwen2-7B


# Training Arguments
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune-qwen2
RUN_NAME=qwen2-7B_moka-ave-r-4-alpha-16-blc_weight-0.25_7_25_12:06
OUTP_DIR=results
OUTPUT_LOG=results/log/qwen2/ave/moka-r-4-alpha-16-qwen2_7b-blc_weight-0.25_7_25_12:06.log

export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'

torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/finetune_qwen2-moka.py \
    --deepspeed deepspeed/stage1-offload.json \
    --llm_name qwen2-7b \
    --reserved_modality None \
    --loramethod train \
    --model_name_or_path $qwen2_ckpt_path \
    --exp_desc "baseline" \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --blc_weight 0.25 \
    --blc_alpha 1 \
    --bf16 False \
    --tf32 True \
    --fp16 False \
    --avqa_task False \
    --ave_task True \
    --save_modules vl_projector,al_projector,lora \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path /nfs1/outdated/WYT/models/clip-vit-large-patch14 \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /nfs1/outdated/WYT/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir $OUTP_DIR/$WANDB_PROJECT/$RUN_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ddp_find_unused_parameters True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --dataloader_num_workers 4 \
    --report_to tensorboard >> "${OUTPUT_LOG}" 2>&1
