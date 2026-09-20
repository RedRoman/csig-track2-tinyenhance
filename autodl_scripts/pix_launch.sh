#!/bin/bash
cd /cache/zihaohe/csig_isr
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf CSIG_ROOT=/cache/zihaohe/csig_isr TINYSR_ROOT=/cache/zihaohe/csig_isr/TinySR PYTHONPATH= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== smoke ==="
CUDA_VISIBLE_DEVICES=3 ./venv/bin/python /cache/zihaohe/train_pix.py --steps 60 --bs 2 --iqa_batch 1 --val_every 30 --val_crops 3 --iqa_start 10 --gan_start 20 --workers 4 --out /cache/zihaohe/csig_isr/train_out/pix_smoke 2>&1 | grep -E "PixUNet|蒸馏样本|^\[|val |done|Error|Traceback|line "
grep -q "done\." <(CUDA_VISIBLE_DEVICES=3 true; tail -1 /dev/null) || true
if ls /cache/zihaohe/csig_isr/train_out/pix_smoke/best.pth >/dev/null 2>&1; then
  echo "SMOKE_OK -> launch 3 runs"
  nohup env CUDA_VISIBLE_DEVICES=0 ./venv/bin/python /cache/zihaohe/train_pix.py --arch 128,256,512 --nb 3 --ps 4 --steps 30000 --bs 8 --out /cache/zihaohe/csig_isr/train_out/pixA > /cache/zihaohe/pixA.log 2>&1 &
  nohup env CUDA_VISIBLE_DEVICES=1 ./venv/bin/python /cache/zihaohe/train_pix.py --arch 64,128,256 --nb 3 --ps 2 --steps 30000 --bs 8 --out /cache/zihaohe/csig_isr/train_out/pixB > /cache/zihaohe/pixB.log 2>&1 &
  nohup env CUDA_VISIBLE_DEVICES=2 ./venv/bin/python /cache/zihaohe/train_pix.py --arch 128,256,512 --nb 3 --ps 4 --steps 30000 --bs 8 --lr 1e-4 --iqa_start 8000 --gan_start 10000 --out /cache/zihaohe/csig_isr/train_out/pixC > /cache/zihaohe/pixC.log 2>&1 &
  sleep 5; echo "launched: $(pgrep -fc "[t]rain_pix.py")"
else
  echo "SMOKE_FAILED"
fi
