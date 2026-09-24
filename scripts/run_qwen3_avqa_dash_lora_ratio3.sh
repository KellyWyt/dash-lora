#!/bin/bash
set -e

cd /nfs1/outdated/WYT/MokA-copy
bash scripts/finetune/ft_qwen3-8b-avqa-dash-lora-ratio3.sh
bash scripts/infer/infer_qwen3-8b-avqa-dash-lora-ratio3.sh
