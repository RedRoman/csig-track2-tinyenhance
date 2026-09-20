#!/bin/bash
# 在 autodl 上: 装权重 -> runner 渲 output_dir -> 从 zip 解包冒烟 -> zip
set -u
S=/cache/zihaohe/csig_isr/submit_pixL2c; T=/cache/zihaohe/csig_isr/train_out
cd /cache/zihaohe/csig_isr
export TORCHDYNAMO_DISABLE= PYTHONPATH= HF_ENDPOINT=https://hf-mirror.com
true
cat > /tmp/render_pix.py <<'PY'
import os, sys, glob, time, importlib.util, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
MD, OUT, TEST = sys.argv[1], sys.argv[2], "/cache/zihaohe/csig_isr/data/赛题二/测试集"
os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location("rpix", os.path.join(MD, "runner.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0 = time.time(); r = m.Runner(MD); print(f"init {time.time()-t0:.0f}s path={r._path}", flush=True)
x = (torch.rand(1,3,512,512,device="cuda",dtype=torch.float16)*2-1)
with torch.no_grad():
    for _ in range(30): r.infer(x)
    torch.cuda.synchronize(); ts=[]
    for _ in range(200):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); r.infer(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
print(f"512 median {float(np.median(ts)):.3f} ms", flush=True)
files = sorted(glob.glob(os.path.join(TEST, "*.jpg")), key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
for i, f in enumerate(files, 1):
    a = np.asarray(Image.open(f).convert("RGB"), np.float32)/255.
    xi = torch.from_numpy(a).permute(2,0,1)[None].cuda().half()*2-1
    y = r.infer(xi).float().clamp(-1,1)
    Image.fromarray(((y[0].permute(1,2,0).cpu().numpy()*0.5+0.5)*255).round().astype(np.uint8)).save(os.path.join(OUT, os.path.basename(f)), quality=97, subsampling=0)
    if i % 25 == 0: print(f"[{i}/100]", flush=True)
print("RENDER_OK", flush=True)
PY
CUDA_VISIBLE_DEVICES=3 ./venv/bin/python /tmp/render_pix.py $S/model_dir $S/output_dir 2>&1 | grep -E "init|median|\[|RENDER_OK|Error|Traceback|line "
cd $S && find model_dir -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; rm -f *.zip; zip -q -r "赛题二_TinyEnhance_干就完了_18020768360.zip" output_dir model_dir -x "*.pyc" && echo "ZIP_MD5 $(md5sum *.zip | cut -c1-8) files=$(unzip -l *.zip | tail -1 | awk '{print $2}') bytes=$(stat -c%s *.zip)"
rm -rf /tmp/zcpixL2c && mkdir -p /tmp/zcpixL2c && cd /tmp/zcpixL2c && unzip -q $S/*.zip model_dir/\* && cd /cache/zihaohe/csig_isr && CUDA_VISIBLE_DEVICES=3 ./venv/bin/python - <<'PY'
import importlib.util, torch, time, numpy as np
spec = importlib.util.spec_from_file_location("rz", "/tmp/zcpixL2c/model_dir/runner.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0=time.time(); r = m.Runner("/tmp/zcpixL2c/model_dir"); print(f"[zip] init {time.time()-t0:.0f}s path={r._path}")
big=(torch.rand(1,3,2160,3840,device="cuda",dtype=torch.float16)*2-1); y=r.infer(big); print("[zip] 4K", tuple(y.shape), "1080p", tuple(r.infer(big[:,:,:1080,:1920]).shape), "odd", tuple(r.infer(big[:,:,:1001,:1503]).shape))
print("ZIPCHECK_OK")
PY
