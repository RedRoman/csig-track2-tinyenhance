#!/bin/bash
# 30674 -> AutoDL 迁移。多流并行绕开单流带宽上限。
R="rsync -a --partial --info=stats2 --no-inc-recursive --exclude=__pycache__"
LOG=/cache/zihaohe/_push; mkdir -p $LOG
cd /cache/zihaohe

run(){ n=$1; shift; ( $R "$@" > $LOG/$n.log 2>&1; echo "=== $n DONE rc=$? $(date +%H:%M:%S) ===" >> $LOG/$n.log ) & }

# 第一批：模型跑起来必需的
run s1_weights   VOSR_weights hf                       autodl:/cache/zihaohe/
run s2_tinysr    csig_isr/TinySR csig_isr/v3 csig_isr/data  autodl:/cache/zihaohe/csig_isr/
run s3_venv      csig_isr/venv                         autodl:/cache/zihaohe/csig_isr/
run s4_sd21base  csig_isr/sd21base                     autodl:/cache/zihaohe/csig_isr/
wait
echo "===== 第一批完成 $(date) =====" >> $LOG/_batch1.done

# 第二批：训练数据与历史 ckpt（排除两个仍在写的 run）
run s5_traindata csig_isr/train_data                   autodl:/cache/zihaohe/csig_isr/
run s6_trainout  --exclude=mosaic_cap/ --exclude=vosr_cap/ csig_isr/train_out autodl:/cache/zihaohe/csig_isr/
wait
echo "===== 第二批完成 $(date) =====" >> $LOG/_batch2.done
