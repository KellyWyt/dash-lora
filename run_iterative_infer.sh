#! /bin/bash
# 定义日志文件路径
LOG_FILE="run_iterative_infer.log"

# 清空之前的日志文件
> "$LOG_FILE"

# 记录脚本开始运行的时间
START_TIME=$(date +%s)
echo "[$(date)] run_iterative_infer.sh started." >> "$LOG_FILE"

# 定义处理中断的函数
handle_shutdown() {
    # 记录脚本被中断的时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$(awk "BEGIN {printf \"%.2f\", $DURATION / 3600}")
    echo "[$(date)] run_iterative_infer.sh interrupted. Duration: $HOURS hours." >> "$LOG_FILE"
    
    # 终止所有子进程
    kill $PID1 $PID2 $PID3 2>/dev/null
    exit 1
}

# 捕获 SIGINT 和 SIGTERM 信号
trap handle_shutdown SIGINT SIGTERM

# 运行第一个脚本
nohup bash scripts/finetune/infer_wSeed-42.sh > logs/[$(date +%Y%m%d)]_MokA_Llama2-7b-chat-hf_seed42.log 2>&1 &
PID1=$!

# 等待第一个脚本执行完毕
wait $PID1
echo "[$(date)] First script completed." >> "$LOG_FILE"

nohup bash scripts/finetune/infer_wSeed-123.sh > logs/[$(date +%Y%m%d)]_MokA_Llama2-7b-chat-hf_seed123.log 2>&1 &
PID1=$!

# 等待第二个脚本执行完毕
wait $PID1
echo "[$(date)] Second script completed." >> "$LOG_FILE"

nohup bash scripts/finetune/infer_wSeed-456.sh > logs/[$(date +%Y%m%d)]_MokA_Llama2-7b-chat-hf_seed456.log 2>&1 &
PID1=$!

# 等待第三个脚本执行完毕
wait $PID1
echo "[$(date)] Third script completed." >> "$LOG_FILE"

# 记录脚本结束运行的时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
# HOURS=$(echo "scale=2; $DURATION / 3600" | bc)
# echo "[$(date)] run_multi_task.sh completed. Duration: $HOURS hours." >> "$LOG_FILE"
# [type awk] for check whether there is a command named 'awk'
HOURS=$(awk "BEGIN {printf \"%.2f\", $DURATION / 3600}")
echo "[$(date)] run_iterative_infer.sh completed. Duration: $HOURS hours." >> "$LOG_FILE"