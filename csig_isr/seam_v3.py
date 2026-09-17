"""接缝检测 v3：用绝对差值而非比值，避免平坦区（分母趋零）导致的假阳性。

v2 的错误：ratio = line_grad/(neighbor_grad+1e-8)，在天空等平坦区
邻域梯度≈0，比值虚高到上千，把真实的电线/边缘误判成接缝。

v3 判据：seam_excess = mean(|边界线差分| - |两侧差分|)，绝对量纲。
若 tile 融合有问题，边界处会有系统性的额外跳变（正值且显著）。
对照组：把同样的检测跑在 LQ 上（LQ 无 tile 处理，其值代表噪声基线）。
"""
import numpy as np, torch, os, sys, glob
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"; dev=torch.device("cuda")

def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255
    return torch.from_numpy(a).permute(2,0,1)[None].to(dev)

def seam_excess(x, stride, axis):
    """返回各 tile 边界的 excess（边界差分 - 邻域差分的均值，绝对量纲）。"""
    g=x.mean(1,keepdim=True)
    d=((g[:,:,1:,:]-g[:,:,:-1,:]).abs()[0,0] if axis==0
       else (g[:,:,:,1:]-g[:,:,:,:-1]).abs()[0,0])
    n=d.shape[axis]
    out=[]
    for p in range(stride, n-1, stride):
        if p<4 or p+4>=n: continue
        line = d[p,:] if axis==0 else d[:,p]
        nb = (torch.cat([d[p-3:p-1,:],d[p+2:p+4,:]],0).mean(0) if axis==0
              else torch.cat([d[:,p-3:p-1],d[:,p+2:p+4]],1).mean(1))
        out.append((p, float((line-nb).mean())))
    return out

print(f"{'图':<12}{'来源':<26}{'横向最大excess':>16}{'竖向最大excess':>16}")
print("-"*72)
for name in ["case11.jpg","case18.jpg","case69.jpg","case92.jpg"]:
    for tag, d, st in [("LQ(无tile,基线)", f"{ROOT}/data/赛题二/测试集", 384),
                       ("旧版 ov128", f"{ROOT}/submit/output_dir", 384),
                       ("新版 ov192", f"{ROOT}/submit/seamfix", 320)]:
        p=os.path.join(d,name)
        if not os.path.exists(p): continue
        x=load(p)
        h=max(seam_excess(x,st,0),key=lambda t:t[1])[1]
        v=max(seam_excess(x,st,1),key=lambda t:t[1])[1]
        print(f"{name if tag.startswith('LQ') else '':<12}{tag:<26}{h:>16.6f}{v:>16.6f}")
        del x; torch.cuda.empty_cache()
    print()
print("判据：若输出的 excess 与 LQ 基线同量级 -> 无接缝")
print("      若输出显著高于 LQ（比如 >2倍且绝对值 >0.002）-> 存在接缝")
