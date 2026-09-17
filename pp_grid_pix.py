"""后处理网格:在 v8 原始渲染上施加变换,测试集 NR 四项(官方目标图)+ 验证集六项(FR 看门)。
用法: pp_grid.py <cfg_id> [cfg_id ...]"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")
RAW = "/cache/zihaohe/raw_pixB2"
M = create_metrics(DEV)

def gauss_k(r, sigma):
    ax = torch.arange(-r, r+1, dtype=torch.float32)
    g = torch.exp(-ax**2/(2*sigma**2)); g = g/g.sum()
    return (g[:,None]*g[None,:]).to(DEV)
def unsharp(x, r, s):
    if s <= 0: return x
    k = gauss_k(r, max(r/2., 0.8)).expand(3,1,2*r+1,2*r+1)
    b = F.conv2d(F.pad(x,(r,r,r,r),mode="reflect"), k, groups=3)
    return x + s*(x-b)
def apply(x, c):
    x = unsharp(x, c.get("r",1), c.get("s",0.12))
    if "clar" in c:  # 大半径中频清晰度
        x = unsharp(x, 8, c["clar"])
    if "con" in c: x = (x-0.5)*c["con"]+0.5
    if "sat" in c:
        g = (x*torch.tensor([0.299,0.587,0.114],device=DEV).view(1,3,1,1)).sum(1,keepdim=True)
        x = g + c["sat"]*(x-g)
    if "gam" in c: x = x.clamp(1e-6,1)**c["gam"]
    if "noise" in c:
        gen = torch.Generator(device=DEV).manual_seed(0)
        x = x + c["noise"]/255.*torch.randn(x.shape, device=DEV, generator=gen)
    return x.clamp(0,1)

CFGS = {
 0: {"s":0.12},                       # 现役基线
 1: {"s":0.16},
 2: {"s":0.12,"con":1.03}, 3: {"s":0.12,"con":1.06}, 4: {"s":0.12,"con":0.97},
 5: {"s":0.12,"sat":1.05}, 6: {"s":0.12,"sat":1.10}, 7: {"s":0.12,"sat":0.95},
 8: {"s":0.12,"gam":0.97}, 9: {"s":0.12,"gam":1.03},
 10: {"r":2,"s":0.12}, 11: {"r":4,"s":0.10},
 12: {"s":0.12,"clar":0.08}, 13: {"s":0.12,"clar":0.15},
 14: {"s":0.12,"noise":0.5}, 15: {"s":0.12,"noise":1.0},
 16: {"s":0.0},
}
def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None].to(DEV)
PAIRS = [("case1","case1_gt.png"),("case2","case2_gt.png"),("case3","case3_gt.jpg")]
tests = sorted(glob.glob(f"{RAW}/test/*.png"))
for cid in [int(a) for a in sys.argv[1:]]:
    c = CFGS[cid]
    ss = []; fr = {"lpips":[], "dists":[]}
    for name, gtf in PAIRS:
        sr = apply(load(f"{RAW}/val/{name}_lq.png"), c); gt = load(os.path.join(VAL, gtf))
        v = {}
        with torch.autocast("cuda", dtype=torch.float16):
            for k in ("lpips","dists"): v[k] = float(M[k](sr,gt)); fr[k].append(v[k]); torch.cuda.empty_cache()
        for k in ("niqe","musiq","maniqa","clipiqa"): v[k] = float(M[k](sr)); torch.cuda.empty_cache()
        ss.append((1-v["lpips"])+(1-v["dists"])+v["clipiqa"]+v["maniqa"]+v["musiq"]/100+max(0,(10-v["niqe"])/10))
    nr = {k: [] for k in ("niqe","musiq","maniqa","clipiqa")}
    for f in tests:
        sr = apply(load(f), c)
        for k in nr: nr[k].append(float(M[k](sr))); torch.cuda.empty_cache()
    m = {k: float(np.mean(v)) for k, v in nr.items()}
    tot = (10-m["niqe"])/10 + m["musiq"]/100 + m["maniqa"] + m["clipiqa"]
    print(f"CFG {cid:2d} {json.dumps(c):40s} val_S={np.mean(ss):.4f} val_FR={np.mean(fr['lpips'])+np.mean(fr['dists']):.4f} "
          f"test_NR={tot:.4f} niqe={m['niqe']:.3f} musiq={m['musiq']:.2f} maniqa={m['maniqa']:.4f} clipiqa={m['clipiqa']:.4f}", flush=True)
print("PP_DONE", flush=True)
