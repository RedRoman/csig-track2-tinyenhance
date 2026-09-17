"""诊断：训练损失里的 MUSIQ（512 crop 尺度）与官方评测的 MUSIQ（原生 4K）
是否反相关。

背景：训练时 out 是 512 输出，loss_metrics["musiq"](out) 就在这个尺度上算。
而官方评 100 张完整 4K 图。已有一次证据显示两者符号相反
（combo vs run8：crop +10.77，full −4.57）。

若在多个 checkpoint 上确认反相关，则「训练优化 MUSIQ」= 「把最终分往下推」，
这解释了 run12 之后每轮训练都变差（MUSIQ 崩 11-13）。
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
from tinysr_runtime import build_model
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
PAIRS = [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]
CKS = [("run8", f"{R}/train_out/tinysr8_fid/best.pth"),
       ("run11", f"{R}/train_out/run11_big_iqa20/best.pth"),
       ("run12", f"{R}/train_out/run12_iqa30/best.pth"),
       ("ref1", f"{R}/train_out/ref1/best.pth"),
       ("ref2", f"{R}/train_out/ref2/best.pth"),
       ("ref3", f"{R}/train_out/ref3/best.pth")]

def seed_all(s=1234):
    import random
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]

@torch.no_grad()
def tiled(model, x, tile=512, ov=256):
    _,_,H,W = x.shape; st = tile-ov
    ys = list(range(0,max(H-tile,0)+1,st)) or [0]
    xs = list(range(0,max(W-tile,0)+1,st)) or [0]
    if ys[-1]+tile<H: ys.append(max(H-tile,0))
    if xs[-1]+tile<W: xs.append(max(W-tile,0))
    t = torch.linspace(-1,1,tile,device=DEV); g = torch.exp(-(t**2)/(2*0.45**2))
    wm = (g[:,None]*g[None,:])[None,None].float()
    out = torch.zeros(1,3,H,W,device=DEV); acc = torch.zeros(1,1,H,W,device=DEV)
    for y in ys:
        for xx in xs:
            with torch.autocast(device_type="cuda", dtype=DT):
                p = model(x[:,:,y:y+tile,xx:xx+tile].to(DEV,dtype=DT)).float()
            out[:,:,y:y+tile,xx:xx+tile]+=p*wm; acc[:,:,y:y+tile,xx:xx+tile]+=wm
    return (out/acc.clamp_min(1e-8)).clamp(0,1)

M = create_metrics(DEV)
imgs = [load(os.path.join(VAL,l)) for l,_ in PAIRS]
res = {}
for tag, ck in CKS:
    if not os.path.exists(ck): print(f"  缺 {tag}"); continue
    model,_ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                          alpha_mode="fixed", fixed_alpha=1.0)
    model.eval(); seed_all()
    crop_v, full_v = [], []
    for lq in imgs:
        H,W = lq.shape[-2:]
        # 训练口径：512 crop 单次前向
        c = lq[:,:,H//2-256:H//2+256, W//2-256:W//2+256].to(DEV,dtype=DT)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=DT):
                oc = model(c).float().clamp(0,1)
        crop_v.append(float(M["musiq"](oc)))
        # 官方口径：整图 tile 融合
        sr = tiled(model, lq)
        full_v.append(float(M["musiq"](sr)))
        del sr, oc; torch.cuda.empty_cache()
    res[tag] = (float(np.mean(crop_v)), float(np.mean(full_v)))
    print(f"  {tag:<6} crop-MUSIQ {res[tag][0]:6.2f}   full-MUSIQ {res[tag][1]:6.2f}", flush=True)
    del model; torch.cuda.empty_cache()

print()
ks = list(res)
c = np.array([res[k][0] for k in ks]); f = np.array([res[k][1] for k in ks])
print(f"{'checkpoint':<10}{'crop':>9}{'full':>9}")
for k in ks: print(f"{k:<10}{res[k][0]:>9.2f}{res[k][1]:>9.2f}")
if len(ks) > 2:
    r = float(np.corrcoef(c, f)[0,1])
    print(f"\n相关系数 r = {r:+.4f}")
    print("  r<0 => 训练优化 crop-MUSIQ 会压低 full-MUSIQ，训练目标与评测错位")
json.dump(res, open(f"{R}/eval_out/musiq_scale.json","w"), indent=2)
