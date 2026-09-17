"""分析赛题验证集的退化特性：LQ vs GT 差异归因。"""
import os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

D = "/cache/zihaohe/csig_isr/data/赛题二/验证集"

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

def radial_spectrum(gray):
    """返回归一化径向频谱能量曲线。"""
    f = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    nb = min(cy, cx)
    prof = np.bincount(r.ravel(), f.ravel(), minlength=nb + 1)[:nb]
    cnt = np.bincount(r.ravel(), minlength=nb + 1)[:nb]
    return prof / np.maximum(cnt, 1)

pairs = [("case1_lq.jpg", "case1_gt.png"),
         ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]

for lqn, gtn in pairs:
    lq = load(os.path.join(D, lqn))
    gt = load(os.path.join(D, gtn))
    print(f"===== {lqn}  vs  {gtn} =====")
    print(f"  shape  LQ={lq.shape}  GT={gt.shape}")
    if lq.shape != gt.shape:
        print("  尺寸不一致，跳过")
        continue

    print(f"  PSNR(LQ,GT)      = {psnr(lq, gt):.3f} dB")
    diff = lq - gt
    print(f"  mean diff        = {diff.mean():+.3f}   (亮度偏移)")
    print(f"  std  diff        = {diff.std():.3f}")
    print(f"  per-channel mean = R{diff[:,:,0].mean():+.2f} G{diff[:,:,1].mean():+.2f} B{diff[:,:,2].mean():+.2f}")
    print(f"  LQ  mean/std     = {lq.mean():.2f} / {lq.std():.2f}")
    print(f"  GT  mean/std     = {gt.mean():.2f} / {gt.std():.2f}")

    # 中心 1024 crop 做频谱与梯度分析
    h, w, _ = lq.shape
    cy, cx = h // 2, w // 2
    s = 512
    lc = lq[cy-s:cy+s, cx-s:cx+s].mean(axis=2)
    gc = gt[cy-s:cy+s, cx-s:cx+s].mean(axis=2)

    # 梯度能量（清晰度代理）
    def grad_energy(g):
        gx = np.diff(g, axis=1); gy = np.diff(g, axis=0)
        return np.sqrt((gx**2).mean() + (gy**2).mean())
    print(f"  梯度能量 LQ/GT   = {grad_energy(lc):.3f} / {grad_energy(gc):.3f}  "
          f"(比值 {grad_energy(lc)/max(grad_energy(gc),1e-6):.3f})")

    # 拉普拉斯方差（模糊程度代理）
    def lapvar(g):
        k = g[1:-1,1:-1]*4 - g[:-2,1:-1] - g[2:,1:-1] - g[1:-1,:-2] - g[1:-1,2:]
        return k.var()
    print(f"  Laplacian var    = LQ {lapvar(lc):.2f} / GT {lapvar(gc):.2f}  "
          f"(比值 {lapvar(lc)/max(lapvar(gc),1e-6):.3f})")

    # 频谱：高频保留比
    sl = radial_spectrum(lc); sg = radial_spectrum(gc)
    n = len(sl)
    for name, lo, hi in [("低频 0-10%", 0, n//10),
                         ("中频 10-40%", n//10, int(n*0.4)),
                         ("高频 40-100%", int(n*0.4), n)]:
        ratio = sl[lo:hi].sum() / max(sg[lo:hi].sum(), 1e-9)
        print(f"  {name:14s} LQ/GT 能量比 = {ratio:.4f}")

    # 噪声估计：高频残差的标准差
    def noise_est(g):
        from itertools import product
        m = (g[:-1,:-1] - g[1:,:-1] - g[:-1,1:] + g[1:,1:])
        return m.std() / 2.0
    print(f"  噪声估计 LQ/GT   = {noise_est(lc):.3f} / {noise_est(gc):.3f}")
    print()
