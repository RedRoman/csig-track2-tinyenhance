"""渲染路径质量旋钮扫描(b 协议口径,官方现跑语义):
baseline / hflip x2 / dihedral x4 / dihedral x8 / CAS 0.1 / CAS 0.2 / x2+CAS0.1
TTA 只用无插值变换(翻转+90k 旋转),tile 级应用。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F
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

# 无插值二面体群:k 旋转 + 可选水平翻转
def d8_fwd(x, i):
    if i >= 4: x = torch.flip(x, [-1])
    return torch.rot90(x, i % 4, [-2, -1])
def d8_inv(x, i):
    x = torch.rot90(x, -(i % 4), [-2, -1])
    if i >= 4: x = torch.flip(x, [-1])
    return x

def cas(x, s):
    """unsharp: x + s*(x-blur3(x)),融合后整图应用"""
    k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]], device=x.device)/16.
    k = k.expand(3,1,3,3)
    b = F.conv2d(F.pad(x, (1,1,1,1), mode="reflect"), k, groups=3)
    return (x + s*(x-b)).clamp(0,1)

@torch.no_grad()
def tiled(model, x01, tta_ids=(0,), sharpen=0.0):
    tile, ov = 512, 256; _,_,H,W = x01.shape; st = tile-ov
    ys = list(range(0,max(H-tile,0)+1,st)) or [0]
    xs = list(range(0,max(W-tile,0)+1,st)) or [0]
    if ys[-1]+tile<H: ys.append(max(H-tile,0))
    if xs[-1]+tile<W: xs.append(max(W-tile,0))
    t = torch.linspace(-1,1,tile,device=DEV); g = torch.exp(-(t**2)/(2*0.45**2))
    wm = (g[:,None]*g[None,:])[None,None].float()
    out = torch.zeros(1,3,H,W,device=DEV); acc = torch.zeros(1,1,H,W,device=DEV)
    for y in ys:
        for xx in xs:
            patch = x01[:,:,y:y+tile,xx:xx+tile].to(DEV,dtype=DT)
            preds = []
            for i in tta_ids:
                with torch.autocast("cuda", dtype=DT):
                    p = model(d8_fwd(patch, i))
                preds.append(d8_inv(p.float(), i))
            p = torch.stack(preds).mean(0)
            out[:,:,y:y+tile,xx:xx+tile] += p*wm; acc[:,:,y:y+tile,xx:xx+tile] += wm
    sr = (out/acc.clamp_min(1e-8)).clamp(0,1)
    if sharpen > 0: sr = cas(sr, sharpen)
    return sr

model, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth",
                       device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
model.eval()
CFGS = [("cas_0.05", (0,), 0.05), ("cas_0.08", (0,), 0.08), ("cas_0.12", (0,), 0.12),
        ("cas_0.15", (0,), 0.15)]
for tag, ids, sh in CFGS:
    ss, comps = [], []
    for lqf, gtf in PAIRS:
        lq = load(os.path.join(VAL,lqf)); gt = load(os.path.join(VAL,gtf)).to(DEV)
        sr = tiled(model, lq, ids, sh)
        v = {}
        with torch.autocast("cuda", dtype=torch.float16):
            for k in ("lpips","dists"):
                v[k] = float(M[k](sr,gt)); torch.cuda.empty_cache()
        for k in ("niqe","musiq","maniqa","clipiqa"):
            v[k] = float(M[k](sr)); torch.cuda.empty_cache()
        ss.append((1-v["lpips"])+(1-v["dists"])+v["clipiqa"]+v["maniqa"]+v["musiq"]/100+max(0,(10-v["niqe"])/10))
        comps.append(v)
    mm = {k: np.mean([c[k] for c in comps]) for k in comps[0]}
    print(f"{tag:14s} S={np.mean(ss):.4f}  lpips={mm['lpips']:.4f} dists={mm['dists']:.4f} "
          f"niqe={mm['niqe']:.3f} musiq={mm['musiq']:.2f} maniqa={mm['maniqa']:.4f} clipiqa={mm['clipiqa']:.4f}", flush=True)
print("TTA_SWEEP_DONE", flush=True)
