#!/bin/bash


# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=4    # NOTE: change here for training with $NPROC_PER_NODE GPUs
MASTER_PORT=6666
RANK=0

llama_ckpt_path=/nfs1/WYT/MokA/pretrained_ckpts/Llama-2-7b-chat-hf
OUTPUT_LOG=results/finetune-share_1.28_17:49_random_ratio_1.log

# Training Arguments
LOCAL_BATCH_SIZE=4 # 6
GRADIENT_ACCUMULATION_STEPS=3 # 2
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune-share-1-15
RUN_NAME=llama_music-share-56-ratio-1-random
OUTP_DIR=results
# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'





########### Arguments ###########
#################################

# lora_r: 444 means three LoRA_A with rank $4$. If there are two modalities, please set as "44".
# blc_weight: control the weight of cross-attention ouput, range [0,1].
# save_modules: set trainable modules, e.g., vl_projector, al_projector, lora.
# vit_ckpt_path: the path of the pre-trained ViT checkpoint. NOTE: change here
# BEATs_ckpt_path: the path of the pre-trained BEATs checkpoint. NOTE: change here
# avqa_task: set True to train on AVQA task

#################################






torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/finetune.py \
    --deepspeed deepspeed/stage1-offload.json \
    --llm_name llama \
    --reserved_modality None \
    --loramethod train \
    --model_name_or_path $llama_ckpt_path \
    --exp_desc "baseline" \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 444 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --top_k_layers 56 \
    --ratio 1 \
    --blc_weight 1 \
    --blc_alpha 1 \
    --bf16 False \
    --tf32 True \
    --fp16 False \
    --avqa_task True \
    --ave_task False \
    --save_modules vl_projector,al_projector,lora \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path /nfs1/WYT/MokA/pretrained_ckpts/clip-vit-large-patch14 \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /nfs1/WYT/MokA/pretrained_ckpts/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
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