"""查清 full-MUSIQ 崩在哪个尺度。

三个互斥假设：
  H1 局部纹理  —— 模型生成的编织纹在原生分辨率上就难看，MUSIQ 直接罚
  H2 拼接伪影  —— tile 融合在 4K 上留下低频接缝/块状，单块 512 前向看不到
  H3 尺度错位  —— 输出在 512 上好看，但 MUSIQ 对 4K 做大幅下采样后，
                  被放大的是别的东西（对比度/色偏/块状），与 512 观感无关

判别：
  A. 直接 512 前向  vs  整图 tile 后再裁同一位置的 512
     两者差很大 => H2
  B. 整图输出在 4K / 2K / 1K / 512 各尺度上的 MUSIQ 曲线
     run8 与 ref1 的排序在哪一档翻转，就是问题所在的尺度
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
from tinysr_runtime import build_model
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
PAIRS = [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]
CKS = [("run8", f"{R}/train_out/tinysr8_fid/best.pth"),
       ("run12", f"{R}/train_out/run12_iqa30/best.pth"),
       ("ref1",  f"{R}/train_out/ref1/best.pth")]

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
musiq = lambda t: float(M["musiq"](t))
imgs = [load(os.path.join(VAL,l)) for l,_ in PAIRS]
gts  = [load(os.path.join(VAL,g)) for _,g in PAIRS]

SCALES = [1, 2, 4, 8]   # 下采样倍数

def scale_curve(t):
    out = []
    for s in SCALES:
        u = t if s==1 else F.interpolate(t, scale_factor=1.0/s, mode="bicubic",
                                         align_corners=False, antialias=True).clamp(0,1)
        out.append(musiq(u))
    return out

rows = {}
# 先给 LQ / GT 打锚
for tag, src in (("LQ", imgs), ("GT", gts)):
    seed_all(); acc = np.zeros(len(SCALES))
    for t in src: acc += np.array(scale_curve(t.to(DEV)))
    rows[tag] = {"scales": (acc/len(src)).tolist(), "crop": None, "tiledcrop": None}
    print(f"  {tag:<6} scales " + " ".join(f"{v:6.2f}" for v in rows[tag]["scales"]), flush=True)

for tag, ck in CKS:
    if not os.path.exists(ck): print(f"  缺 {tag}"); continue
    model,_ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                          alpha_mode="fixed", fixed_alpha=1.0)
    model.eval(); seed_all()
    acc = np.zeros(len(SCALES)); cd = []; ct = []
    for lq in imgs:
        H,W = lq.shape[-2:]; y0,x0 = H//2-256, W//2-256
        # A1 直接 512 前向
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DT):
            oc = model(lq[:,:,y0:y0+512, x0:x0+512].to(DEV,dtype=DT)).float().clamp(0,1)
        cd.append(musiq(oc))
        # A2 整图 tile 后裁同一位置
        sr = tiled(model, lq)
        ct.append(musiq(sr[:,:,y0:y0+512, x0:x0+512].contiguous()))
        # B  尺度曲线
        acc += np.array(scale_curve(sr))
        del sr, oc; torch.cuda.empty_cache()
    rows[tag] = {"scales": (acc/len(imgs)).tolist(),
                 "crop": float(np.mean(cd)), "tiledcrop": float(np.mean(ct))}
    print(f"  {tag:<6} scales " + " ".join(f"{v:6.2f}" for v in rows[tag]["scales"])
          + f"   direct512 {rows[tag]['crop']:6.2f}  tiled@512 {rows[tag]['tiledcrop']:6.2f}", flush=True)
    del model; torch.cuda.empty_cache()

print("\n=== MUSIQ 随下采样倍数 ===")
print(f"{'':8}" + "".join(f"{'/'+str(s):>9}" for s in SCALES) + f"{'direct512':>11}{'tiled@512':>11}")
for k,v in rows.items():
    line = f"{k:<8}" + "".join(f"{x:>9.2f}" for x in v["scales"])
    line += f"{v['crop']:>11.2f}" if v["crop"] is not None else f"{'-':>11}"
    line += f"{v['tiledcrop']:>11.2f}" if v["tiledcrop"] is not None else f"{'-':>11}"
    print(line)
json.dump(rows, open(f"{R}/eval_out/musiq_where.json","w"), indent=2)
