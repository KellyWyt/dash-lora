#!/bin/bash

# ================= 配置区域 =================
# 目标脚本路径 (根据你的实际路径调整)
TARGET_SCRIPT="scripts/finetune/ft_qwen2_moka.sh"

# 显存阈�? (MiB)
# 注意：即使空闲，显存也不一定是0，通常几百M，所以设 1000 比较安全
THRESHOLD=3000 

# 检查间�? (�?)
INTERVAL=20
# ===========================================

echo "开始蹲�?... 等待所�? GPU 显存低于 $THRESHOLD MiB"

while true; do
    # 获取所有显卡的显存使用�?
    # 输出格式示例:
    # 400

    # 10
    # 24000
    # 10
    MEM_USAGES=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    
    ALL_FREE=true
    
    # 遍历每一张卡的显�?
    for mem in $MEM_USAGES; do
        # 只要有一张卡大于阈值，就标记为不空�?
        if [ "$mem" -gt "$THRESHOLD" ]; then
            ALL_FREE=false
            break
        fi
    done

    # 判断结果
    if [ "$ALL_FREE" = true ]; then
        echo "=========================================="
        echo "[$(date)] 发现空闲！所�? GPU 显存均低�? $THRESHOLD MiB"
        echo "正在启动任务: $TARGET_SCRIPT"
        echo "=========================================="
        
        # === 核心：执行你的脚�? ===
        bash "$TARGET_SCRIPT"
        
        # 执行完退出循环，不再监控
        echo "[$(date)] 任务执行完毕�?"
        break 
    else
        echo "[$(date)] GPU 忙碌�?... $INTERVAL 秒后再检�?"
        # 打印一下当前的显存情况给用户看
        # echo "$MEM_USAGES" | tr '\n' '\t' ; echo "" # 可选：打印具体数�?
        sleep $INTERVAL
    fi
done
