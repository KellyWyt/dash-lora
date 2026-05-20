nohup bash scripts/finetune/infer_with_MA.sh \
    > [$(date +%Y%m%d_%H%M%S)]_memLoRA_missingAudio.log 2>&1 &
PID1=$!
wait $PID1
nohup bash scripts/finetune/infer_with_MV.sh \
    > [$(date +%Y%m%d_%H%M%S)]_memLoRA_missingVideo.log 2>&1 &
PID1=$!
wait $PID1

