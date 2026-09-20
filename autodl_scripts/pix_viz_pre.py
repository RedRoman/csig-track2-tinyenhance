import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code"); sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch, torch.nn.functional as F, time
from PIL import Image
from settings import VAL
from pix_model import build_pix
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
m = build_pix("/cache/zihaohe/csig_isr/train_out/pixC/best.pth", device=DEV, dtype=DT)
def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]
os.makedirs("/cache/zihaohe/viz_pre", exist_ok=True)
for c in ("case1","case2","case3"):
    lq = load(f"{VAL}/{c}_lq.jpg"); gt = load(f"{VAL}/{c}_gt.png" if c!="case3" else f"{VAL}/{c}_gt.jpg")
    with torch.no_grad(), torch.autocast("cuda", dtype=DT):
        torch.cuda.synchronize(); t0=time.time(); sr = m(lq.to(DEV, DT)).float().clamp(0,1); torch.cuda.synchronize(); dt=time.time()-t0
    v8 = load(f"/cache/zihaohe/raw_v8/val/{c}_lq.png")
    lo = F.avg_pool2d(sr.cpu(), 16); ll = F.avg_pool2d(lq, 16); d = (lo-ll).abs().mean(1).flatten()
    lf_pix = float(torch.topk(d, max(1,int(d.numel()*0.001))).values.mean())
    lo8 = F.avg_pool2d(v8, 16); d8 = (lo8-ll).abs().mean(1).flatten()
    lf_v8 = float(torch.topk(d8, max(1,int(d8.numel()*0.001))).values.mean())
    print(f"{c}: 4K整图推理 {dt*1000:.0f}ms  lf_top pixB2={lf_pix:.4f} v8={lf_v8:.4f}  size={tuple(lq.shape[-2:])}", flush=True)
    H, W = lq.shape[-2:]
    for tag, (y, x) in {"A": (H//3, W//3), "B": (H//2, W//5), "C": (H*2//3, W*3//5)}.items():
        y = min(y, H-512); x = min(x, W-512)
        row = torch.cat([lq[0,:,y:y+512,x:x+512], sr[0].cpu()[:,y:y+512,x:x+512], v8[0,:,y:y+512,x:x+512], gt[0,:,y:y+512,x:x+512]], 2)
        Image.fromarray((row.permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(f"/cache/zihaohe/viz_pre/{c}_{tag}.png")
print("VIZ_DONE", flush=True)
