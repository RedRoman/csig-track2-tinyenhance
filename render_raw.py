"""v8 原始 tiled 渲染(无 CAS)落盘为无损 PNG:100 测试 + 3 验证。后处理网格在此基础上做。"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from settings import VAL, TEST
from tinysr_runtime import build_model
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
OUT = "/cache/zihaohe/raw_v8"
model, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8_best.pth",
                       device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
model.eval()
def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]
@torch.no_grad()
def tiled(x01, tile=512, ov=256):
    _,_,H,W = x01.shape; st = tile-ov
    ys = list(range(0,max(H-tile,0)+1,st)) or [0]; xs = list(range(0,max(W-tile,0)+1,st)) or [0]
    if ys[-1]+tile<H: ys.append(max(H-tile,0))
    if xs[-1]+tile<W: xs.append(max(W-tile,0))
    t = torch.linspace(-1,1,tile,device=DEV); g = torch.exp(-(t**2)/(2*0.45**2))
    wm = (g[:,None]*g[None,:])[None,None].float()
    out = torch.zeros(1,3,H,W,device=DEV); acc = torch.zeros(1,1,H,W,device=DEV)
    for y in ys:
        for xx in xs:
            with torch.autocast("cuda", dtype=DT):
                p = model(x01[:,:,y:y+tile,xx:xx+tile].to(DEV,dtype=DT)).float()
            out[:,:,y:y+tile,xx:xx+tile] += p*wm; acc[:,:,y:y+tile,xx:xx+tile] += wm
    return (out/acc.clamp_min(1e-8)).clamp(0,1)
def save(t, p):
    Image.fromarray((t[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(p, compress_level=1)
os.makedirs(f"{OUT}/val", exist_ok=True); os.makedirs(f"{OUT}/test", exist_ok=True)
for f in ("case1_lq.jpg","case2_lq.jpg","case3_lq.jpg"):
    save(tiled(load(os.path.join(VAL,f))), f"{OUT}/val/{f[:-4]}.png")
tf = sorted(glob.glob(os.path.join(TEST,"*.jpg")), key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
for i, f in enumerate(tf, 1):
    save(tiled(load(f)), f"{OUT}/test/{os.path.basename(f)[:-4]}.png")
    if i % 20 == 0: print(f"raw {i}/100", flush=True)
print("RAW_DONE", flush=True)
