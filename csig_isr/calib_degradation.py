"""退化标定：用 3 组真实配对反推「过度降噪」退化算子。

目标：找到一个可微/可复现的退化算子 D，使得 D(GT) 在统计上匹配真实 LQ。
匹配指标：径向频谱曲线、Laplacian 方差、梯度能量、噪声水平、PSNR。
"""
import os, json
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None
D = "/cache/zihaohe/csig_isr/data/赛题二/验证集"
OUT = "/cache/zihaohe/csig_isr/degradation_calib"
os.makedirs(OUT, exist_ok=True)

PAIRS = [("case1_lq.jpg", "case1_gt.png"),
         ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def crops(img, size=512, n=6, seed=0):
    """从大图取若干 512 crop（固定种子，保证 LQ/GT 对齐）。"""
    rng = np.random.RandomState(seed)
    h, w, _ = img.shape
    out = []
    for _ in range(n):
        y = rng.randint(0, h - size)
        x = rng.randint(0, w - size)
        out.append((y, x))
    return out


def stats(img):
    """单张 crop 的统计特征。"""
    g = img.mean(axis=2)
    gx = np.diff(g, axis=1); gy = np.diff(g, axis=0)
    grad = np.sqrt((gx ** 2).mean() + (gy ** 2).mean())
    lap = g[1:-1,1:-1]*4 - g[:-2,1:-1] - g[2:,1:-1] - g[1:-1,:-2] - g[1:-1,2:]
    m = (g[:-1,:-1] - g[1:,:-1] - g[:-1,1:] + g[1:,1:])
    return dict(grad=float(grad), lapvar=float(lap.var()), noise=float(m.std()/2.0))


def radial(gray):
    f = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    h, w = f.shape
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y-cy)**2 + (x-cx)**2).astype(np.int32)
    nb = min(cy, cx)
    prof = np.bincount(r.ravel(), f.ravel(), minlength=nb+1)[:nb]
    cnt = np.bincount(r.ravel(), minlength=nb+1)[:nb]
    return prof / np.maximum(cnt, 1)


def band_ratio(lq, gt):
    """三个频段的能量比 LQ/GT。"""
    sl = radial(lq.mean(axis=2)); sg = radial(gt.mean(axis=2))
    n = len(sl)
    out = {}
    for name, lo, hi in [("low", 0, n//10), ("mid", n//10, int(n*0.4)), ("high", int(n*0.4), n)]:
        out[name] = float(sl[lo:hi].sum() / max(sg[lo:hi].sum(), 1e-9))
    return out


def psnr(a, b):
    mse = np.mean((a-b)**2)
    return float(10*np.log10(255.0**2/mse)) if mse > 0 else 99.0


# ============ 候选退化算子 ============
def deg_bilateral(gt, d=9, sc=75, ss=75):
    out = np.empty_like(gt)
    u8 = np.clip(gt, 0, 255).astype(np.uint8)
    return cv2.bilateralFilter(u8, d, sc, ss).astype(np.float32)


def deg_nlm(gt, h=10):
    u8 = np.clip(gt, 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoisingColored(u8, None, h, h, 7, 21).astype(np.float32)


def deg_guided(gt, r=8, eps=1e-2):
    u8 = np.clip(gt, 0, 255).astype(np.uint8)
    try:
        return cv2.ximgproc.guidedFilter(u8, u8, r, eps*255*255).astype(np.float32)
    except Exception:
        return None


def deg_gauss(gt, sigma=1.0):
    return cv2.GaussianBlur(gt, (0, 0), sigma)


def deg_downup(gt, scale=2, interp=cv2.INTER_CUBIC):
    h, w = gt.shape[:2]
    small = cv2.resize(gt, (w//scale, h//scale), interpolation=interp)
    return cv2.resize(small, (w, h), interpolation=interp)


CANDIDATES = {
    "bilateral_d9_s75":   lambda g: deg_bilateral(g, 9, 75, 75),
    "bilateral_d9_s150":  lambda g: deg_bilateral(g, 9, 150, 150),
    "bilateral_d15_s100": lambda g: deg_bilateral(g, 15, 100, 100),
    "nlm_h10":            lambda g: deg_nlm(g, 10),
    "nlm_h20":            lambda g: deg_nlm(g, 20),
    "guided_r8":          lambda g: deg_guided(g, 8, 1e-2),
    "guided_r16":         lambda g: deg_guided(g, 16, 1e-2),
    "gauss_s1.0":         lambda g: deg_gauss(g, 1.0),
    "gauss_s1.5":         lambda g: deg_gauss(g, 1.5),
    "gauss_s2.0":         lambda g: deg_gauss(g, 2.0),
    "downup_x2":          lambda g: deg_downup(g, 2),
    "downup_x3":          lambda g: deg_downup(g, 3),
    "bilat+gauss":        lambda g: deg_gauss(deg_bilateral(g, 9, 75, 75), 0.8),
}

# ============ 主流程 ============
print("=" * 90)
print("步骤 1：真实 LQ 的统计指纹")
print("=" * 90)

real_fp = []   # 每个 crop 一条真实指纹
crop_store = []

for lqn, gtn in PAIRS:
    lq_full = load(os.path.join(D, lqn))
    gt_full = load(os.path.join(D, gtn))
    if lq_full.shape != gt_full.shape:
        print(f"跳过 {lqn}（尺寸不符）"); continue
    for (y, x) in crops(lq_full, 512, 6, seed=hash(lqn) % 1000):
        lq = lq_full[y:y+512, x:x+512]
        gt = gt_full[y:y+512, x:x+512]
        # 跳过纯色/过暗区域
        if gt.std() < 12:
            continue
        s_lq, s_gt = stats(lq), stats(gt)
        br = band_ratio(lq, gt)
        fp = dict(
            grad_ratio=s_lq["grad"]/max(s_gt["grad"], 1e-6),
            lapvar_ratio=s_lq["lapvar"]/max(s_gt["lapvar"], 1e-6),
            noise_ratio=s_lq["noise"]/max(s_gt["noise"], 1e-6),
            psnr=psnr(lq, gt), **br)
        real_fp.append(fp)
        crop_store.append((gt, lq))

keys = ["grad_ratio", "lapvar_ratio", "noise_ratio", "low", "mid", "high", "psnr"]
real_mean = {k: float(np.mean([f[k] for f in real_fp])) for k in keys}
real_std = {k: float(np.std([f[k] for f in real_fp])) for k in keys}
print(f"有效 crop 数：{len(real_fp)}")
print(f"{'指标':<14}{'均值':>10}{'标准差':>10}")
for k in keys:
    print(f"{k:<14}{real_mean[k]:>10.4f}{real_std[k]:>10.4f}")

print()
print("=" * 90)
print("步骤 2：候选退化算子匹配度（对 GT 施加算子，与真实 LQ 指纹比对）")
print("=" * 90)

# 归一化距离用的尺度
scale = {k: max(real_std[k], abs(real_mean[k]) * 0.1, 1e-3) for k in keys}

results = []
for name, fn in CANDIDATES.items():
    fps = []
    ok = True
    for gt, lq_real in crop_store[:12]:      # 用 12 个 crop 评估，控制耗时
        syn = fn(gt)
        if syn is None:
            ok = False; break
        s_syn, s_gt = stats(syn), stats(gt)
        br = band_ratio(syn, gt)
        fps.append(dict(
            grad_ratio=s_syn["grad"]/max(s_gt["grad"], 1e-6),
            lapvar_ratio=s_syn["lapvar"]/max(s_gt["lapvar"], 1e-6),
            noise_ratio=s_syn["noise"]/max(s_gt["noise"], 1e-6),
            psnr=psnr(syn, gt), **br))
    if not ok:
        print(f"{name:<20} 不可用（缺 ximgproc）"); continue
    syn_mean = {k: float(np.mean([f[k] for f in fps])) for k in keys}
    # 加权距离：频段比和 lapvar 最重要
    w = dict(grad_ratio=1.0, lapvar_ratio=2.0, noise_ratio=1.5,
             low=2.0, mid=1.5, high=2.0, psnr=0.5)
    dist = sum(w[k] * abs(syn_mean[k]-real_mean[k]) / scale[k] for k in keys) / sum(w.values())
    results.append((dist, name, syn_mean))

results.sort()
print(f"{'算子':<20}{'距离':>8}{'lapvar':>9}{'grad':>8}{'noise':>8}{'low':>8}{'mid':>8}{'high':>8}{'psnr':>8}")
print(f"{'[真实 LQ]':<20}{'--':>8}{real_mean['lapvar_ratio']:>9.3f}{real_mean['grad_ratio']:>8.3f}"
      f"{real_mean['noise_ratio']:>8.3f}{real_mean['low']:>8.3f}{real_mean['mid']:>8.3f}"
      f"{real_mean['high']:>8.3f}{real_mean['psnr']:>8.2f}")
print("-" * 90)
for dist, name, m in results:
    print(f"{name:<20}{dist:>8.3f}{m['lapvar_ratio']:>9.3f}{m['grad_ratio']:>8.3f}"
          f"{m['noise_ratio']:>8.3f}{m['low']:>8.3f}{m['mid']:>8.3f}{m['high']:>8.3f}{m['psnr']:>8.2f}")

json.dump({"real_mean": real_mean, "real_std": real_std,
           "candidates": [{"name": n, "dist": d, "stats": m} for d, n, m in results]},
          open(os.path.join(OUT, "calib.json"), "w"), indent=1, ensure_ascii=False)
print(f"\n结果已存 {OUT}/calib.json")
