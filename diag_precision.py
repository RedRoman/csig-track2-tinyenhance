"""精度崩塌诊断:eager vs compile(max-autotune) vs AOTI,单图全尺度,分项六项+互PSNR。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
torch.backends.cudnn.benchmark = False
DEV, DT = torch.device("cuda"), torch.float16
CK = "/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth"
SO = "/cache/zihaohe/csig_isr/submit_v8/tinysr_v8_aoti.so"
Image.MAX_IMAGE_PIXELS = None

from tinysr_runtime import build_model
from settings import VAL
from metrics_utils import create_metrics
core, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
core.eval()
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, t):
        out = s.m(t * 0.5 + 0.5)
        if isinstance(out, tuple): out = out[0]
        return out * 2 - 1
eager = Wrap(core).to(DEV).eval()
aoti = torch._export.aot_load(SO, device="cuda")
M = create_metrics(DEV)

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

def six(sr, gt, tag):
    v = {k: float(M[k](sr,gt) if k in ("lpips","dists") else M[k](sr))
         for k in ("lpips","dists","niqe","musiq","maniqa","clipiqa")}
    sc = (1-v["lpips"])+(1-v["dists"])+v["clipiqa"]+v["maniqa"]+v["musiq"]/100+max(0,(10-v["niqe"])/10)
    print(f"{tag:<10} lpips={v['lpips']:.4f} dists={v['dists']:.4f} niqe={v['niqe']:.3f} "
          f"musiq={v['musiq']:.2f} maniqa={v['maniqa']:.4f} clipiqa={v['clipiqa']:.4f} Score={sc:.4f}", flush=True)
    return sr

lq = load(os.path.join(VAL, "case1_lq.jpg")); gt = load(os.path.join(VAL, "case1_gt.png")).to(DEV)
se = six(tiled(lambda t: eager(t), lq), gt, "eager")
sa = six(tiled(lambda t: aoti(t), lq), gt, "AOTI")
d = (se-sa).abs()
print(f"渲染图互差 max={float(d.max()):.4f} mean={float(d.mean()):.5f} "
      f"PSNR={10*np.log10(1/max(float((d**2).mean()),1e-12)):.2f}", flush=True)
for im, name in ((se,"diag_eager.png"),(sa,"diag_aoti.png")):
    arr = (im[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    Image.fromarray(arr).save(f"/cache/zihaohe/{name}")
# 单 tile 逐区间分布:差异集中在何值域
with torch.no_grad():
    xx = (lq[:, :, :512, :512].to(DEV,dtype=DT)*2-1)
    ye = eager(xx).float(); ya = aoti(xx).float()
    dd = (ye-ya).abs()
    print(f"单tile |d| p50={dd.median():.4f} p99={dd.flatten().kthvalue(int(dd.numel()*0.99)).values:.4f} max={dd.max():.4f}", flush=True)
print("DIAG_DONE", flush=True)
