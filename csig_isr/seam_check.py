"""严格验证接缝是否真实存在。

判据：真接缝在整条 tile 边界上系统性出现（高比例的行/列都突变），
      而图像真实边缘只在局部出现。用"突变像素占该边界的比例"区分。
"""
import numpy as np, torch, os
from PIL import Image
import torch.nn.functional as F
Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"
dev=torch.device("cuda")
TILE,OV=512,128; STRIDE=TILE-OV

def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255
    return torch.from_numpy(a).permute(2,0,1)[None].to(dev)

def seam_profile(x, axis):
    """返回每条 tile 边界上'突变像素占比'。"""
    g=x.mean(1,keepdim=True)
    if axis==0:  # 横向边界（沿 y）
        d=(g[:,:,1:,:]-g[:,:,:-1,:]).abs()[0,0]
        n=d.shape[0]
    else:
        d=(g[:,:,:,1:]-g[:,:,:,:-1]).abs()[0,0]
        n=d.shape[1]
    out=[]
    for p in range(STRIDE, n-1, STRIDE):
        if p<4 or p+4>=n: continue
        line = d[p,:] if axis==0 else d[:,p]
        nb = (d[p-4:p-1,:].mean(0)+d[p+2:p+5,:].mean(0))/2 if axis==0 \
             else (d[:,p-4:p-1].mean(1)+d[:,p+2:p+5].mean(1))/2
        ratio = line/(nb+1e-8)
        frac = float((ratio>1.5).float().mean())   # 突变像素占比
        out.append((p, float(ratio.mean()), frac))
    return out

for name in ["case18.jpg","case69.jpg","case92.jpg","case11.jpg"]:
    sr=load(f"{ROOT}/submit/output_dir/{name}")
    lq=load(f"{ROOT}/data/赛题二/测试集/{name}")
    print(f"=== {name} ===")
    for axis,lab in [(0,"横向"),(1,"竖向")]:
        ps=seam_profile(sr,axis); pl=seam_profile(lq,axis)
        worst=max(ps,key=lambda t:t[2])
        # 同位置 LQ 的占比作为对照（LQ 无 tile 处理，其值代表图像本身边缘）
        lqf=[t[2] for t in pl if t[0]==worst[0]]
        print(f"  {lab}最强边界 pos={worst[0]:>5}  均值比{worst[1]:.3f}  "
              f"突变占比 输出{worst[2]*100:.1f}%  LQ对照{(lqf[0]*100 if lqf else 0):.1f}%")
    del sr,lq; torch.cuda.empty_cache()
print()
print("判据：若'输出突变占比'远高于'LQ对照'，说明是真接缝；接近则是图像本身边缘")
