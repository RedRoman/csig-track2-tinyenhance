#!/bin/bash
unset PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
B=https://hf-mirror.com
R=/cache/zihaohe/csig_isr/bases
mkdir -p $R

dl(){ # repo, file, dest
  mkdir -p "$(dirname "$3")"
  [ -f "$3" ] && [ "$(stat -c%s "$3")" -gt 100000 ] && { echo "  [跳过] $2"; return; }
  curl -sL --retry 3 -C - --max-time 2400 -o "$3" "$B/$1/resolve/main/$2"
  echo "  [完成] $2  $(du -h "$3" 2>/dev/null|cut -f1)"
}

echo "=== PiSA-SR (可调保真/语义双 LoRA，最适配) ==="
for f in lora/pisa_pix.safetensors lora/pisa_sem.safetensors; do
  dl ndtran0101/pisa-sr-diffusers "$f" "$R/pisasr/$f"
done
for f in config.json diffusion_pytorch_model.safetensors; do
  dl ndtran0101/pisa-sr-diffusers "unet_pix/$f" "$R/pisasr/unet_pix/$f"
done

echo "=== S3Diff ==="
for f in s3diff.pkl de_net.pth; do dl zhangap/S3Diff "$f" "$R/s3diff/$f"; done

echo "=== FiDeSR (CVPR2026) ==="
dl jmjin2/FiDeSR fidesr.pkl "$R/fidesr/fidesr.pkl"

echo "=== 清单 ==="
find $R -type f -size +1M -exec ls -lh {} \; | awk '{print $5, $9}'
echo "BASES_DONE"
