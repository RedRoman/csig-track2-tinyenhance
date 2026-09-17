"""诊断：Score 六项各剩多少空间 + DiT 计算构成分解。

关键问题：
  1. GT 自身的六项指标是多少？（天花板参考，尤其 NIQE）
  2. 我们与 GT 每项差多少？哪项性价比最高？
  3. 测试集 100 张 4K 的无参考指标（无 GT 但 4 项可算，占 Score 58%）
  4. DiT 的 token 数/层数对时延的贡献
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
VAL = f"{ROOT}/data/赛题二/验证集"
TEST = f"{ROOT}/data/赛题二/测试集"
dev = torch.device("cuda")
PAIRS = [("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
         ("case3_lq.jpg","case3_gt.jpg",303)]

def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1)[None]

import pyiqa
NR = {"niqe":pyiqa.create_metric("niqe",device=dev),
      "musiq":pyiqa.create_metric("musiq",device=dev),
      "maniqa":pyiqa.create_metric("maniqa-pipal",device=dev),
      "clipiqa":pyiqa.create_metric("clipiqa",device=dev)}

print("="*78)
print("1. GT 自身的无参考指标（Score 的天花板参考）")
print("="*78)
gt_crops=[]
for lqn, gtn, seed in PAIRS:
    gt_f = load_img(f"{VAL}/{gtn}")
    rng = np.random.RandomState(seed); _,_,H,W = gt_f.shape
    got,tries=0,0
    while got<6 and tries<72:
        tries+=1
        y,x = rng.randint(0,H-512), rng.randint(0,W-512)
        c = gt_f[:,:,y:y+512,x:x+512]
        if c.std().item()<0.045: continue
        got+=1; gt_crops.append(c)
res={k:[] for k in NR}
for c in gt_crops:
    c=c.to(dev)
    for k,f in NR.items(): res[k].append(f(c).item())
    torch.cuda.empty_cache()
g={k:float(np.mean(v)) for k,v in res.items()}
print(f"  GT: niqe {g['niqe']:.4f}  musiq {g['musiq']:.2f}  maniqa {g['maniqa']:.4f}  clipiqa {g['clipiqa']:.4f}")
gscore_nr = g['clipiqa']+g['maniqa']+g['musiq']/100+max(0,(10-g['niqe'])/10)
print(f"  GT 的无参考四项贡献 = {gscore_nr:.4f}  (若有参考两项满分则 Score 上限 = {2+gscore_nr:.4f})")

print()
print("="*78)
print("2. 各项差距分析（TinySR run2 @1200 vs GT）")
print("="*78)
ours = {"lpips":0.3471,"dists":0.2775,"clipiqa":0.6123,"maniqa":0.4783,"musiq":57.2994,"niqe":7.2546}
print(f"  {'指标':<10}{'我们':>10}{'GT':>10}{'当前贡献':>10}{'GT贡献':>10}{'可提升':>10}")
print("  "+"-"*60)
rows=[("clipiqa",ours['clipiqa'],g['clipiqa'],ours['clipiqa'],g['clipiqa']),
      ("maniqa",ours['maniqa'],g['maniqa'],ours['maniqa'],g['maniqa']),
      ("musiq",ours['musiq'],g['musiq'],ours['musiq']/100,g['musiq']/100),
      ("niqe",ours['niqe'],g['niqe'],max(0,(10-ours['niqe'])/10),max(0,(10-g['niqe'])/10))]
tot=0
for k,ov,gv,oc,gc in rows:
    d=gc-oc; tot+=max(d,0)
    print(f"  {k:<10}{ov:>10.4f}{gv:>10.4f}{oc:>10.4f}{gc:>10.4f}{d:>+10.4f}")
print(f"  {'有参考两项':<10}{'':<10}{'':<10}{(1-ours['lpips'])+(1-ours['dists']):>10.4f}{2.0:>10.4f}{2.0-((1-ours['lpips'])+(1-ours['dists'])):>+10.4f}")
print(f"\n  无参考四项若追平 GT 可得 {tot:+.4f} 分")

print()
print("="*78)
print("3. 测试集 100 张 4K 的 LQ 无参考指标（真实分布，占 Score 58%）")
print("="*78)
import glob
tf = sorted(glob.glob(f"{TEST}/*.jpg"))[:20]
res={k:[] for k in NR}
for p in tf:
    im = load_img(p)
    _,_,H,W = im.shape
    c = im[:,:,H//2-256:H//2+256, W//2-256:W//2+256].to(dev)
    for k,f in NR.items(): res[k].append(f(c).item())
    torch.cuda.empty_cache()
t={k:float(np.mean(v)) for k,v in res.items()}
print(f"  测试集 LQ (20张中心crop): niqe {t['niqe']:.4f}  musiq {t['musiq']:.2f}  "
      f"maniqa {t['maniqa']:.4f}  clipiqa {t['clipiqa']:.4f}")
print(f"  验证集 LQ 对比:          niqe 9.8828  musiq 30.63  maniqa 0.3175  clipiqa 0.2474")
json.dump({"gt":g,"test_lq":t}, open(f"{ROOT}/eval_out/diagnose.json","w"), indent=1)
