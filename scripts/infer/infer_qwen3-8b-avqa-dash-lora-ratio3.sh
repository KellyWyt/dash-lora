#!/bin/bash

NPROC_PER_NODE=4
MASTER_PORT=6672
qwen3_ckpt_path=pretrained_ckpts/Qwen3-8B
YOUR_CKPT_PATH=results/finetune-qwen3/qwen3-8B_avqa_dash-lora-r-4-alpha-16-ratio-3/checkpoint-2994

export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
    scripts/infer/inference_qwen_3.py \
    --seed 42 --mode test --llm_name qwen3-8b \
    --reserved_modality None --loratype dash-lora --loramethod test \
    --model_name_or_path "$qwen3_ckpt_path" \
    --freeze_backbone True --lora_enable True --bits 32 \
    --lora_r 4 --lora_alpha 16 --lora_dropout 0.05 \
    --top_k_layers 65 --ratio 3 --dash_lora_safe_importance True \
    --blc_weight 1 --blc_alpha 1 \
    --bf16 True --tf32 False --fp16 False \
    --ckpt_dir "$YOUR_CKPT_PATH" \
    --avqa_task True --ave_task False \
    --visual_branch True --video_frame_nums 10 \
    --vit_ckpt_path /nfs1/outdated/WYT/models/clip-vit-large-patch14 \
    --image_size 224 --patch_size 14 --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /nfs1/outdated/WYT/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 --output_dir not_used \
    >> "$YOUR_CKPT_PATH/inference_avqa.log" 2>&1
