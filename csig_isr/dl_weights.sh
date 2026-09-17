#!/bin/bash
# 下载 AdcSR 全套权重与标准测试集
unset PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
BASE=https://hf-mirror.com/Guaishou74851/AdcSR/resolve/main
cd /cache/zihaohe/csig_isr/AdcSR || exit 1
mkdir -p weight/pretrained testset

FILES="
weight/net_params_200.pkl
weight/pretrained/DAPE.pth
weight/pretrained/halfDecoder.ckpt
weight/pretrained/osediff.pkl
weight/pretrained/ram_swin_large_14m.pth
testset/DIV2K-Val.zip
testset/DRealSR.zip
testset/RealSR.zip
"

for f in $FILES; do
    echo "=== downloading $f"
    curl -L --retry 5 --retry-delay 3 -C - --max-time 3600 \
         -o "$f" "$BASE/$f" 2>&1 | tail -1
    ls -lh "$f"
done

echo "=== 解压测试集 ==="
cd testset || exit 1
for z in DIV2K-Val.zip DRealSR.zip RealSR.zip; do
    [ -f "$z" ] && unzip -oq "$z" && echo "unzipped $z"
done
cd ..

echo "=== 最终清单 ==="
find weight testset -maxdepth 2 -type f -size +1M -exec ls -lh {} \; | awk '{print $5, $9}'
echo "=== 测试集图片数 ==="
for d in testset/*/; do
    [ -d "$d" ] && echo "$d: $(find "$d" -name '*.png' | wc -l) png"
done
echo "ALL_DONE"
