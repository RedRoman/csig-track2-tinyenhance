#!/bin/bash
# 20:25 定时清场:杀掉本用户的 train_pix/eval_pix/time_pair 进程(含 dataloader worker),留日志。
T=$(date -d "today 20:25" +%s); N=$(date +%s); S=$((T-N)); [ $S -lt 0 ] && S=0
echo "watchdog armed at $(date), sleeping $S s until 20:25" > /cache/zihaohe/pix_runs/watchdog.log
sleep $S
{
  echo "=== fire $(date)"
  for n in pixN1 pixN2 pixL2e; do echo "-- $n last step: $(grep -o '^\[[0-9]*/' /cache/zihaohe/pix_runs/$n.log | tail -1)"; done
  pkill -TERM -u "$(id -u)" -f "[t]rain_pix.py|[e]val_pix.py|[t]ime_pair.py" && echo "TERM sent"
  sleep 20
  pkill -KILL -u "$(id -u)" -f "[t]rain_pix.py|[e]val_pix.py|[t]ime_pair.py" && echo "KILL sent (stragglers)"
  sleep 5; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "=== done $(date)"
} >> /cache/zihaohe/pix_runs/watchdog.log 2>&1
