"""崩塌根因二分:模型 fp16 敏感度基线 vs inductor 特有偏移;逐子模块定位。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
torch.backends.cudnn.benchmark = False
DEV = torch.device("cuda")
CK = "/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth"
from tinysr_runtime import build_model
from settings import VAL
Image.MAX_IMAGE_PIXELS = None

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]
x01 = load(os.path.join(VAL, "case1_lq.jpg"))[:, :, :512, :512].to(DEV)

# A. fp16 vs fp32 eager(模型自身精度敏感度)
core16, _ = build_model(CK, device=DEV, dtype=torch.float16, merge_lora=True, alpha_mode="auto")
core16.eval()
with torch.no_grad():
    y16 = core16(x01.half())
    if isinstance(y16, tuple): y16 = y16[0]
core32, _ = build_model(CK, device=DEV, dtype=torch.float32, merge_lora=True, alpha_mode="auto")
core32.eval()
with torch.no_grad():
    y32 = core32(x01.float())
    if isinstance(y32, tuple): y32 = y32[0]
d = (y16.float()-y32).abs()
print(f"A. eager fp16 vs fp32: p50={d.median():.4f} max={d.max():.4f}", flush=True)

# B. compile(default) fp16 vs eager fp16(inductor 偏移,非 AOTI)
cm = torch.compile(core16, mode="default")
with torch.no_grad():
    yc = cm(x01.half())
    if isinstance(yc, tuple): yc = yc[0]
d2 = (yc.float()-y16.float()).abs()
print(f"B. compile-default fp16 vs eager fp16: p50={d2.median():.4f} max={d2.max():.4f}", flush=True)

# C. alpha 读数对比(gate 是否行为不同)
with torch.no_grad():
    out16 = core16(x01.half())
    a16 = float(out16[1].mean()) if isinstance(out16, tuple) else float('nan')
print(f"C. eager alpha={a16}", flush=True)
print("DIAG2_DONE", flush=True)
