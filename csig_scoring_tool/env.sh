# source 本文件后再跑工具。把权重与数据指到本包;不依赖网络。
export CSIG_TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="$CSIG_TOOL/weights/torch"
export CSIG_DATA_ROOT="$CSIG_TOOL/data"
export CSIG_MANIQA_SAMPLES=100
unset PYTHONPATH
# CLIP 主干 pyiqa 固定读 ~/.cache/clip/RN50.pt;csig_metrics.ensure_weights() 会自动软链,这里再兜一层
if [ -f "$CSIG_TOOL/weights/clip/RN50.pt" ]; then
    mkdir -p ~/.cache/clip
    [ -e ~/.cache/clip/RN50.pt ] || ln -s "$CSIG_TOOL/weights/clip/RN50.pt" ~/.cache/clip/RN50.pt
else
    echo "Missing scoring weights: restore csig_scoring_tool/weights/ before full evaluation" >&2
fi
echo "CSIG_TOOL=$CSIG_TOOL  TORCH_HOME=$TORCH_HOME"
