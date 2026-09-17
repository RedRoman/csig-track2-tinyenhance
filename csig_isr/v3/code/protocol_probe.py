"""用两次提交的出图反推官方的指标计算协议。

两次提交：同样 100 张图、同样加速比(12.20/12.22ms)、只有模型不同。
加速比项在榜分之比里约掉，所以 72.6047/71.54 = 1.014883 只由 Score 决定：
    线性形式  ΔScore = +0.058
    指数形式  ΔScore = +0.021
逐个协议算 ΔNR，看哪个能落进这个区间。
"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F, pyiqa
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")
A = "/cache/zihaohe/csig_isr/submit/output_dir"        # run8 -> 71.54
B = "/cache/zihaohe/csig_isr/submit_v3/output_dir"     # combo_hf1500 -> 72.6047
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25

def P_full(t):   return t
def P_crop(t):
    _,_,H,W=t.shape; return t[:,:,H//2-256:H//2+256, W//2-256:W//2+256]
def P_sq512(t):  return F.interpolate(t,(512,512),mode="bicubic",align_corners=False).clamp(0,1)
def P_short512(t):
    _,_,H,W=t.shape; s=512/min(H,W)
    return F.interpolate(t,(int(round(H*s)),int(round(W*s))),mode="bicubic",align_corners=False).clamp(0,1)
def P_long1024(t):
    _,_,H,W=t.shape; s=1024/max(H,W)
    return F.interpolate(t,(int(round(H*s)),int(round(W*s))),mode="bicubic",align_corners=False).clamp(0,1)
def P_half(t):
    _,_,H,W=t.shape
    return F.interpolate(t,(H//2,W//2),mode="bicubic",align_corners=False).clamp(0,1)
def P_tilegrid(t):   # 3x3 均匀取 512 块后平均（非只取中心）
    _,_,H,W=t.shape; out=[]
    for y in np.linspace(0,H-512,3).astype(int):
        for x in np.linspace(0,W-512,3).astype(int):
            out.append(t[:,:,y:y+512,x:x+512])
    return out
PROTOS = [("full 原生4K",P_full),("center512",P_crop),("squash512",P_sq512),
          ("short=512",P_short512),("long=1024",P_long1024),("half 2K",P_half),
          ("3x3 tile 均值",P_tilegrid)]
M = {k: pyiqa.create_metric(k, device=DEV) for k in ("niqe","musiq","maniqa-pipal","clipiqa")}
names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(A,"*.jpg")))[:N]

def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]

acc={pn:{d:{k:[] for k in M} for d in ("A","B")} for pn,_ in PROTOS}
for i,nm in enumerate(names):
    for dtag,d in (("A",A),("B",B)):
        p=os.path.join(d,nm)
        if not os.path.exists(p): continue
        t=load(p)
        for pn,fn in PROTOS:
            xs=fn(t); xs=xs if isinstance(xs,list) else [xs]
            for k,mf in M.items():
                with torch.no_grad():
                    try: acc[pn][dtag][k].append(float(np.mean([float(mf(x.to(DEV))) for x in xs])))
                    except Exception: pass
        torch.cuda.empty_cache()
    if (i+1)%5==0: print(f"  {i+1}/{len(names)}",flush=True)

def nrsum(m):
    return m["clipiqa"]+m["maniqa-pipal"]+m["musiq"]/100+max(0,(10-m["niqe"])/10)
print()
print(f"{'协议':<16}{'ΔNIQE':>9}{'ΔMUSIQ':>9}{'ΔMANIQA':>10}{'ΔCLIPIQA':>10}{'ΔNR和':>9}{'ΔScore预估':>12}")
print("-"*76)
DFR = -0.093   # 全尺度验证图实测的 lpips+dists 项变化
out={}
for pn,_ in PROTOS:
    mA={k:float(np.mean(v)) for k,v in acc[pn]["A"].items() if v}
    mB={k:float(np.mean(v)) for k,v in acc[pn]["B"].items() if v}
    if len(mA)<4 or len(mB)<4: continue
    d={k:mB[k]-mA[k] for k in mA}
    dnr=nrsum(mB)-nrsum(mA)
    out[pn]={"A":mA,"B":mB,"dnr":dnr}
    print(f"{pn:<16}{d['niqe']:>+9.3f}{d['musiq']:>+9.2f}{d['maniqa-pipal']:>+10.4f}"
          f"{d['clipiqa']:>+10.4f}{dnr:>+9.4f}{dnr+DFR:>+12.4f}")
print()
print("需要匹配: 线性形式 +0.058 / 指数形式 +0.021")
json.dump(out,open("/cache/zihaohe/csig_isr/eval_out/protocol_probe.json","w"),indent=2)
