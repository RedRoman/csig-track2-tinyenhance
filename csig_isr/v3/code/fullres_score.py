"""在【全分辨率验证图】上算完整六项 Score —— 这才是与官方口径一致的评测。

验证集 3 组是全分辨率 LQ/GT 配对，所以 LPIPS/DISTS 也能算。
对比 512 crop 口径，看我们一直在优化的代理到底偏了多少。
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
from settings import VAL
from metrics_utils import create_metrics, proxy_score
from tinysr_runtime import build_model
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
T = "/cache/zihaohe/csig_isr/train_out"
PAIRS = [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]
CFGS = [("run8", f"{T}/tinysr8_fid/best.pth"),
        ("combo_hf1500", f"{T}/v3_combo_hf1500/best.pth"),
        ("r8_niqe0.5_hf1500", f"{T}/v3_r8_n0.5_hf1500/best.pth"),
        ("r8_niqe2.0_hf1500", f"{T}/v3_r8_n2.0_hf1500/best.pth"),
        ("sched_h900", f"{T}/v3_sched_h900/best.pth"),
        ("hf_w2250", f"{T}/v3_hf_w2250/final.pth")]

def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]

@torch.no_grad()
def tiled(model, x, tile=512, ov=256):
    _,_,H,W=x.shape; st=tile-ov
    ys=list(range(0,max(H-tile,0)+1,st)) or [0]
    xs=list(range(0,max(W-tile,0)+1,st)) or [0]
    if ys[-1]+tile<H: ys.append(max(H-tile,0))
    if xs[-1]+tile<W: xs.append(max(W-tile,0))
    t=torch.linspace(-1,1,tile,device=DEV); g=torch.exp(-(t**2)/(2*0.45**2))
    wm=(g[:,None]*g[None,:])[None,None].float()
    out=torch.zeros(1,3,H,W,device=DEV); acc=torch.zeros(1,1,H,W,device=DEV)
    for y in ys:
        for xx in xs:
            with torch.autocast(device_type="cuda",dtype=DT):
                p=model(x[:,:,y:y+tile,xx:xx+tile].to(DEV,dtype=DT)).float()
            out[:,:,y:y+tile,xx:xx+tile]+=p*wm; acc[:,:,y:y+tile,xx:xx+tile]+=wm
    return (out/acc.clamp_min(1e-8)).clamp(0,1)

M = create_metrics(DEV, include_psnr=True)
res={}
for tag, ck in CFGS:
    rows_full, rows_crop = [], []
    model,_=build_model(ck,device=DEV,dtype=DT,merge_lora=False,alpha_mode="fixed",fixed_alpha=1.0)
    model.eval()
    for lqn,gtn in PAIRS:
        lq, gt = load(os.path.join(VAL,lqn)), load(os.path.join(VAL,gtn))
        if lq.shape!=gt.shape: continue
        sr = tiled(model, lq)
        gtd = gt.to(DEV)
        for tagd, a, b in (("full", sr, gtd),):
            r={}
            for k in ("lpips","dists"): r[k]=float(M[k](a,b))
            for k in ("niqe","musiq","maniqa","clipiqa"): r[k]=float(M[k](a))
            r["psnr"]=float(M["psnr"](a,b)); rows_full.append(r)
        H,W=gt.shape[-2:]
        cs, cg = sr[:,:,H//2-256:H//2+256,W//2-256:W//2+256], gtd[:,:,H//2-256:H//2+256,W//2-256:W//2+256]
        r={}
        for k in ("lpips","dists"): r[k]=float(M[k](cs,cg))
        for k in ("niqe","musiq","maniqa","clipiqa"): r[k]=float(M[k](cs))
        r["psnr"]=float(M["psnr"](cs,cg)); rows_crop.append(r)
        del sr; torch.cuda.empty_cache()
    del model; torch.cuda.empty_cache()
    agg=lambda rs:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]}
    res[tag]={"full":agg(rows_full),"crop":agg(rows_crop)}
    for s in ("full","crop"):
        res[tag][s]["score"]=proxy_score(res[tag][s])
    print(f"{tag} 完成", flush=True)

K=["lpips","dists","niqe","musiq","maniqa","clipiqa","psnr","score"]
print()
print(f"{'尺度':<6}{'候选':<16}" + ''.join(f'{k:>10}' for k in K))
print("-"*98)
for s in ("crop","full"):
    for tag,_ in CFGS:
        r=res[tag][s]; print(f"{s:<6}{tag:<16}" + ''.join(f'{r[k]:>10.4f}' for k in K))
print()
for s in ("crop","full"):
    d=res["combo_hf1500"][s]["score"]-res["run8"][s]["score"]
    print(f"{s:>4} 口径下 combo_hf1500 相对 run8 的 Score 增益: {d:+.4f}")
json.dump(res, open("/cache/zihaohe/csig_isr/eval_out/fullres_score.json","w"), indent=2)
