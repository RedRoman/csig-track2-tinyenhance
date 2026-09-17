"""AOTI 导出→加载→测速→数值全链验证 + compile 数值对全尺度 Score 的影响。"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
torch.backends.cudnn.benchmark = False

DEV, DT = torch.device("cuda"), torch.float16
CK = "/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth"
PKG = "/cache/zihaohe/csig_isr/submit_v8/tinysr_v8_aoti.pt2"
os.makedirs(os.path.dirname(PKG), exist_ok=True)
Image.MAX_IMAGE_PIXELS = None

def bench(fn, x, n=100, w=20):
    with torch.no_grad():
        for _ in range(w): fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts))

from tinysr_runtime import build_model
core, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
core.eval()
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, t):
        out = s.m(t * 0.5 + 0.5)
        if isinstance(out, tuple): out = out[0]
        return out * 2 - 1
eager = Wrap(core).to(DEV).eval()

torch.manual_seed(3)
x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)
with torch.no_grad(): y_e = eager(x.clone())
t_e = bench(lambda t: eager(t), x)
print(f"eager {t_e:.3f} ms", flush=True)

# AOTI 导出
t0 = time.time()
with torch.no_grad():
    path = torch._export.aot_compile(
        eager, (x,),
        options={"max_autotune": True,
                 "aot_inductor.output_path": PKG.replace(".pt2", ".so")})
print(f"AOTI 导出 {time.time()-t0:.0f}s -> {path}", flush=True)

# 新进程语义的加载(同进程简化验证,跨进程由 runner 冒烟另测)
aoti = torch._export.aot_load(path, device="cuda")
with torch.no_grad(): y_a = aoti(x.clone())
t_a = bench(lambda t: aoti(t), x)
print(f"AOTI {t_a:.3f} ms  提速vs eager {t_e/t_a:.3f}x  |d|={float((y_a-y_e).abs().max()):.2e}", flush=True)

# Score 影响:3 张验证图 tiled 渲染 eager vs AOTI
from settings import VAL
from metrics_utils import create_metrics
import torch.nn.functional as F
M = create_metrics(DEV)
PAIRS = [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]
def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]
@torch.no_grad()
def tiled(fn, x01, tile=512, ov=256):
    _,_,H,W = x01.shape; st = tile-ov
    ys = list(range(0,max(H-tile,0)+1,st)) or [0]
    xs = list(range(0,max(W-tile,0)+1,st)) or [0]
    if ys[-1]+tile<H: ys.append(max(H-tile,0))
    if xs[-1]+tile<W: xs.append(max(W-tile,0))
    t = torch.linspace(-1,1,tile,device=DEV); g = torch.exp(-(t**2)/(2*0.45**2))
    wm = (g[:,None]*g[None,:])[None,None].float()
    out = torch.zeros(1,3,H,W,device=DEV); acc = torch.zeros(1,1,H,W,device=DEV)
    for yy in ys:
        for xx in xs:
            p = fn((x01[:,:,yy:yy+tile,xx:xx+tile].to(DEV,dtype=DT)*2-1)).float()*0.5+0.5
            out[:,:,yy:yy+tile,xx:xx+tile] += p*wm; acc[:,:,yy:yy+tile,xx:xx+tile] += wm
    return (out/acc.clamp_min(1e-8)).clamp(0,1)
def six(sr, gt):
    v = {k: float(M[k](sr,gt) if k in ("lpips","dists") else M[k](sr))
         for k in ("lpips","dists","niqe","musiq","maniqa","clipiqa")}
    return (1-v["lpips"])+(1-v["dists"])+v["clipiqa"]+v["maniqa"]+v["musiq"]/100+max(0,(10-v["niqe"])/10)
se, sa = [], []
for lqf, gtf in PAIRS:
    lq = load(os.path.join(VAL,lqf)); gt = load(os.path.join(VAL,gtf)).to(DEV)
    se.append(six(tiled(lambda t: eager(t), lq), gt))
    sa.append(six(tiled(lambda t: aoti(t), lq), gt))
print(f"全尺度 Score eager={np.mean(se):.4f}  AOTI={np.mean(sa):.4f}  差={np.mean(sa)-np.mean(se):+.4f}", flush=True)
print("VERIFY2_DONE", flush=True)
