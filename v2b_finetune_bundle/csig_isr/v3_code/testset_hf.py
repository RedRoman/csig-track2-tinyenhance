"""在【100 张测试图】上量伪影，不依赖 GT。

做法：以 LQ 自身的局部高频能量定义"本该平坦"的区域（最低 35% 分位），
在这些区域里量各候选输出的高频能量。LQ 平坦 = 信息已被抹平，
输出在这里的高频只能是编造的。

这和 hf_diagnostics.py 的区别：那个用 GT 定义平坦区，因此只能在
3 张验证图上跑；这个用 LQ，可以直接在真正要提交的 100 张上跑。
"""
import os, sys, glob
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

LQ = "/cache/zihaohe/csig_isr/data/赛题二/测试集"
CANDS = [
    ("run8 已提交",      "/cache/zihaohe/csig_isr/submit/output_dir"),
    ("soup1.80 旧候选",  "/cache/zihaohe/csig_isr/submit_v2/output_dir"),
    ("combo_hf1500 待交","/cache/zihaohe/csig_isr/submit_v3/output_dir"),
]
K, Q = 9, 0.35     # 局部窗口、平坦区分位

def hp_energy(a):
    """局部高频能量图：亮度 - 高斯模糊后的亮度，取绝对值再局部平均。"""
    t = torch.from_numpy(a.astype(np.float32)/255.).permute(2,0,1)[None]
    y = (0.299*t[:,0]+0.587*t[:,1]+0.114*t[:,2])[:,None]
    lo = F.avg_pool2d(F.pad(y,(2,2,2,2),mode='reflect'), 5, 1)
    return F.avg_pool2d(F.pad((y-lo).abs(),(K//2,)*4,mode='reflect'), K, 1)

files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(LQ,"*.jpg")))
print(f"测试图 {len(files)} 张，用 LQ 最平坦的 {int(100*Q)}% 像素作为统计区\n")

acc = {n: [] for n,_ in CANDS}
n_ok = 0
for i, nm in enumerate(files):
    lq = np.asarray(Image.open(os.path.join(LQ,nm)).convert("RGB"))
    h,w = lq.shape[:2]
    lq = lq[h//2-768:h//2+768, w//2-768:w//2+768]      # 中央 1536 区域，控制耗时
    if lq.shape[0] < 512: continue
    e_lq = hp_energy(lq)
    thr = torch.quantile(e_lq.flatten(), Q)
    mask = (e_lq <= thr).float()
    if mask.sum() < 100: continue
    ok = True
    row = {}
    for n, d in CANDS:
        p = os.path.join(d, nm)
        if not os.path.exists(p): ok = False; break
        sr = np.asarray(Image.open(p).convert("RGB"))
        sr = sr[h//2-768:h//2+768, w//2-768:w//2+768]
        e_sr = hp_energy(sr)
        row[n] = float((e_sr*mask).sum()/mask.sum()) / max(float((e_lq*mask).sum()/mask.sum()), 1e-9)
    if not ok: continue
    for n,v in row.items(): acc[n].append(v)
    n_ok += 1
    if (i+1) % 25 == 0: print(f"  {i+1}/{len(files)}", flush=True)

print(f"\n有效样本 {n_ok} 张")
print(f"\n{'候选':<22}{'平坦区 SR/LQ 高频比':>22}{'中位数':>10}{'最差图':>10}")
for n,_ in CANDS:
    v = np.array(acc[n])
    if not len(v): print(f"{n:<22}{'(缺图)':>22}"); continue
    print(f"{n:<22}{v.mean():>22.3f}{np.median(v):>10.3f}{v.max():>10.3f}")
