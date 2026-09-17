#!/usr/bin/env bash
# fr_repair（含 w_hf_over=0 的 balanced 对照）+ 按实测梯度标定的 HF 消融
set -uo pipefail
cd "$(dirname "$0")"
echo "################ FR repair ################"
bash run_v3_fr_repair.sh
echo "################ HF ablation（权重按 grad_diagnostics 标定）################"
# hf_over 原始梯度 6.65e-4；消融配置下主梯度(1.5*lpips+2.0*dists)=6.65
#   w=250 -> 约 2.5% ; w=750 -> 约 7.5% ; w=2250 -> 约 22%
W_HF_LIST="250 750 2250" bash run_v3_hf_ablation.sh
echo ALLDONE
