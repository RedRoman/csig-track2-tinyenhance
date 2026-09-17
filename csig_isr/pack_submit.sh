#!/bin/bash
set -e
unset PYTHONPATH
cd /cache/zihaohe/csig_isr/submit

NAME="赛题二_TinyEnhance_干就完了_18020768360"
rm -f "$NAME.zip"

# 赛题样例结构：output_dir 与 model_dir 直接位于 zip 根下，不套外层目录
STAGE=$(mktemp -d)
cp -r output_dir "$STAGE/"
mkdir -p "$STAGE/model_dir"
cp model_dir/tinysr_x1.pt "$STAGE/model_dir/"
cp model_dir/runner.py    "$STAGE/model_dir/"
cp model_dir/config.yaml  "$STAGE/model_dir/"

echo "=== 打包前结构 ==="
find "$STAGE" -maxdepth 1 -mindepth 1 -type d -printf "  %f\n"
echo "  output_dir 图片数: $(ls "$STAGE/output_dir"/*.jpg | wc -l)"
echo "  model_dir 内容:"; ls -lh "$STAGE/model_dir" | awk 'NR>1{print "    "$5, $9}'

echo "=== 压缩（output_dir 与 model_dir 位于 zip 根）==="
( cd "$STAGE" && zip -rq "/cache/zihaohe/csig_isr/submit/$NAME.zip" output_dir model_dir \
    -x "*.DS_Store" -x "__MACOSX/*" )
SZ=$(stat -c%s "$NAME.zip")
echo "  $NAME.zip  $(du -h "$NAME.zip" | cut -f1)  ($SZ bytes)"
if [ "$SZ" -gt 10737418240 ]; then echo "  警告：超过 10GB 上限"; else echo "  大小合规（<=10GB）"; fi

echo "=== 校验 zip 内部结构 ==="
unzip -l "$NAME.zip" | head -8
echo "  ..."
unzip -l "$NAME.zip" | grep -cE "output_dir/case[0-9]+\.jpg" | xargs echo "  output_dir 图片数:"
unzip -l "$NAME.zip" | grep -E "model_dir/" | awk '{print "  ", $4}'
rm -rf "$STAGE"
echo "PACK_DONE"
