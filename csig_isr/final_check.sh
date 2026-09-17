#!/bin/bash
unset PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf
cd /cache/zihaohe/csig_isr/submit
Z="赛题二_TinyEnhance_干就完了_18020768360.zip"
T=$(mktemp -d)
echo "解压到 $T"
unzip -q "$Z" -d "$T"
echo "=== 解压后结构 ==="
find "$T" -maxdepth 1 -mindepth 1 | sed 's#.*/#  #'
echo "  output_dir: $(ls $T/output_dir/*.jpg|wc -l) 张"
echo "  model_dir : $(ls $T/model_dir|tr '\n' ' ')"

echo "=== 用解压出的包跑 Runner ==="
cd /cache/zihaohe/csig_isr
venv/bin/python - "$T/model_dir" <<'PY'
import sys, os, time, importlib.util
import numpy as np, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
MD = sys.argv[1]
spec = importlib.util.spec_from_file_location("r_", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r = m.Runner(MD)
print(f"  Runner 加载成功 device={r.device} dtype={r.weight_dtype}")

a = np.asarray(Image.open("/cache/zihaohe/csig_isr/data/赛题二/测试集/case1.jpg")
               .convert("RGB"), dtype=np.float32)/255.0
t = torch.from_numpy(a).permute(2,0,1)[None][:,:,1000:1512,1000:1512]
x = (t*2-1).to(r.device, dtype=r.weight_dtype)
o = r.infer(x)
print(f"  infer: {tuple(x.shape)} -> {tuple(o.shape)}  range [{o.min():.3f},{o.max():.3f}]")
assert o.shape == x.shape

with torch.no_grad():
    for _ in range(20): r.infer(x)
    torch.cuda.synchronize()
    s,e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(100): r.infer(x)
    e.record(); torch.cuda.synchronize()
lat = s.elapsed_time(e)/100
print(f"  512x512 时延 {lat:.3f} ms   加速比 {212.768/lat:.2f}x (OSEDiff 212.77ms)")
PY
rm -rf "$T"
echo "FINAL_OK"
