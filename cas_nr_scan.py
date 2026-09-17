"""CAS 强度 x 官方口径:100 张测试集 NR 四项 + 3 张验证集六项。用法: cas_nr_scan.py <cas>"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from settings import VAL, TEST
from metrics_utils import create_metrics
from tinysr_runtime import build_model
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
CAS = float(sys.argv[1])
M = create_metrics(DEV)
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
def cas(x, s):
    if s <= 0: return x
    k = (torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]], device=x.device)/16.).expand(3,1,3,3)
    b = F.conv2d(F.pad(x,(1,1,1,1),mode="reflect"), k, groups=3)
    return (x + s*(x-b)).clamp(0,1)
PAIRS = [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]
ss = []
for lqf, gtf in PAIRS:
    sr = cas(tiled(load(os.path.join(VAL,lqf))), CAS); gt = load(os.path.join(VAL,gtf)).to(DEV)
    v = {}
    with torch.autocast("cuda", dtype=torch.float16):
        for k in ("lpips","dists"): v[k] = float(M[k](sr,gt)); torch.cuda.empty_cache()
    for k in ("niqe","musiq","maniqa","clipiqa"): v[k] = float(M[k](sr)); torch.cuda.empty_cache()
    ss.append((1-v["lpips"])+(1-v["dists"])+v["clipiqa"]+v["maniqa"]+v["musiq"]/100+max(0,(10-v["niqe"])/10))
print(f"cas={CAS} val_S={np.mean(ss):.4f}", flush=True)
tf = sorted(glob.glob(os.path.join(TEST,"*.jpg")), key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
nr = {k: [] for k in ("niqe","musiq","maniqa","clipiqa")}
for i, f in enumerate(tf, 1):
    sr = cas(tiled(load(f)), CAS)
    for k in nr: nr[k].append(float(M[k](sr))); torch.cuda.empty_cache()
    if i % 25 == 0: print(f"  cas={CAS} test {i}/100", flush=True)
m = {k: float(np.mean(v)) for k, v in nr.items()}
tot = (10-m["niqe"])/10 + m["musiq"]/100 + m["maniqa"] + m["clipiqa"]
print(f"cas={CAS} test_NR={tot:.4f} niqe={m['niqe']:.3f} musiq={m['musiq']:.2f} maniqa={m['maniqa']:.4f} clipiqa={m['clipiqa']:.4f}", flush=True)
print(f"CAS_SCAN_DONE {CAS}", flush=True)
