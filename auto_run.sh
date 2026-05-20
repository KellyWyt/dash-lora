#!/bin/bash

# --- Configuration ---
INTERVAL=60
# 根据你的日志，GPU 目前占用约 2415MB。
# 如果你认为 2.4GB 算是“空闲”（比如只是驱动占用），请把阈值设为 2500。
# 如果你想等它完全释放到 100MB 以下，请维持 100。
THRESHOLD=2500
TRAIN_SCRIPT="scripts/finetune/ft_qwen2_moka.sh"
MONITOR_LOG="auto_monitor.log"

echo "GPU Monitor Started. Checking every ${INTERVAL}s..." >> $MONITOR_LOG

while true
do
    # 提取显存使用量数字
    usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    usage_arr=($usage)
    
    if [ ${#usage_arr[@]} -lt 4 ]; then
        echo "$(date): Error - Less than 4 GPUs detected." >> $MONITOR_LOG
    else
        gpu0=${usage_arr[0]}
        gpu1=${usage_arr[1]}
        gpu2=${usage_arr[2]}
        gpu3=${usage_arr[3]}

        echo "$(date): Current GPU Usage: [$gpu0, $gpu1, $gpu2, $gpu3] MiB" >> $MONITOR_LOG

        # 检查是否全部低于阈值
        if [ $gpu0 -lt $THRESHOLD ] && [ $gpu1 -lt $THRESHOLD ] && [ $gpu2 -lt $THRESHOLD ] && [ $gpu3 -lt $THRESHOLD ]; then
            echo "$(date): All 4 GPUs are IDLE. Launching training..." >> $MONITOR_LOG
            cd /nfs1/WYT/MokA-copy
            # 启动微调脚本
            bash $TRAIN_SCRIPT
            echo "$(date): Task finished. Monitor exiting." >> $MONITOR_LOG
            break 
        fi
    fi
    sleep $INTERVAL
done