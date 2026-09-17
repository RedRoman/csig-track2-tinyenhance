import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from tinysr_runtime import build_model
core, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth",
                      device=torch.device("cuda"), dtype=torch.float16, merge_lora=True, alpha_mode="auto")
core.eval()
Image.MAX_IMAGE_PIXELS = None
a = np.asarray(Image.open("/cache/zihaohe/csig_isr/data/赛题二/验证集/case1_lq.jpg").convert("RGB"), np.float32)/255.
lq = torch.from_numpy(a).permute(2,0,1)[None][:, :, :512, :512].cuda()
with torch.no_grad():
    y = core(lq.half())
    if isinstance(y, tuple): y = y[0]
y = (y.float()*2-1).cpu()
d = (torch.load("/cache/zihaohe/pair_runner.pt") - y).abs()
print(f"真图 runner vs eager: p50={d.median():.5f} p99={d.flatten().kthvalue(int(d.numel()*0.99)).values:.5f} max={d.max():.4f}", flush=True)
print("B_DONE", flush=True)
