"""用提交 runner 本身渲染 output_dir——评测端重跑 runner 时逐位一致(仅 JPEG 有损差)。"""
import os, sys, glob, time, importlib.util, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
MD = "/cache/zihaohe/csig_isr/submit_v10b/model_dir"
OUT = "/cache/zihaohe/csig_isr/submit_v10b/output_dir"
TEST = "/cache/zihaohe/csig_isr/data/赛题二/测试集"
os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location("rmin", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r = m.Runner(MD)
files = sorted(glob.glob(os.path.join(TEST, "*.jpg")),
               key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
t0 = time.time()
for i, f in enumerate(files, 1):
    a = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.
    x = torch.from_numpy(a).permute(2, 0, 1)[None].cuda().half() * 2 - 1
    y = r.infer(x).float().clamp(-1, 1)
    arr = ((y[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.).round().astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, os.path.basename(f)), quality=97, subsampling=0)
    if i % 10 == 0:
        print(f"[{i}/100] {(time.time()-t0)/i:.1f}s/img", flush=True)
print("RENDER_V10B_DONE", flush=True)
