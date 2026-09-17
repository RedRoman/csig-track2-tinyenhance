"""渲染端 overlap 扫描:v8,3 张验证图全尺度六项。E1/E2 已证 overlap 融合在赚 FR。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
M = create_metrics(DEV)
PAIRS = [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]

@torch.no_grad()
def tiled(model, x01, ov, sigma=0.45, tile=512):
    _,_,H,W = x01.shape; st = tile-ov
    ys = list(range(0,max(H-tile,0)+1,st)) or [0]
    xs = list(range(0,max(W-tile,0)+1,st)) or [0]
    if ys[-1]+tile<H: ys.append(max(H-tile,0))
    if xs[-1]+tile<W: xs.append(max(W-tile,0))
    t = torch.linspace(-1,1,tile,device=DEV); g = torch.exp(-(t**2)/(2*sigma**2))
    wm = (g[:,None]*g[None,:])[None,None].float()
    out = torch.zeros(1,3,H,W,device=DEV); acc = torch.zeros(1,1,H,W,device=DEV)
    for y in ys:
        for xx in xs:
            with torch.autocast("cuda", dtype=DT):
                p = model(x01[:,:,y:y+tile,xx:xx+tile].to(DEV,dtype=DT)).float()
            out[:,:,y:y+tile,xx:xx+tile] += p*wm; acc[:,:,y:y+tile,xx:xx+tile] += wm
    return (out/acc.clamp_min(1e-8)).clamp(0,1)

model, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth",
                       device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
model.eval()
for ov, sigma in ((256,0.45),(320,0.45),(384,0.45),(320,0.6),(384,0.6)):
    scores = []
    for lqf, gtf in PAIRS:
        lq = load(os.path.join(VAL,lqf)); gt = load(os.path.join(VAL,gtf)).to(DEV)
        sr = tiled(model, lq, ov, sigma)
        v = {k: float(M[k](sr,gt) if k in ("lpips","dists") else M[k](sr))
             for k in ("lpips","dists","niqe","musiq","maniqa","clipiqa")}
        scores.append((1-v["lpips"])+(1-v["dists"])+v["clipiqa"]+v["maniqa"]+v["musiq"]/100+max(0,(10-v["niqe"])/10))
    print(f"ov={ov} sigma={sigma}: Score={np.mean(scores):.4f}  ({['%.4f'%s for s in scores]})", flush=True)
print("OV_DONE", flush=True)
