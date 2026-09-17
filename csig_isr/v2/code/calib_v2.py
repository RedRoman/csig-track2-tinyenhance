"""退化标定 v2：加入 JPEG 压缩，网格搜索最优退化链。

v1 结论：高斯低通比双边/NLM/引导滤波更接近真实退化，但合成 PSNR 比真实高 6-7dB。
猜想：LQ 全为 .jpg 而 GT 部分为 .png，JPEG 压缩是退化链的一环。
"""
import os, io, json, itertools
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


def stats(img):
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
    sl = radial(lq.mean(axis=2)); sg = radial(gt.mean(axis=2))
    n = len(sl)
    out = {}
    for name, lo, hi in [("low", 0, n//10), ("mid", n//10, int(n*0.4)), ("high", int(n*0.4), n)]:
        out[name] = float(sl[lo:hi].sum() / max(sg[lo:hi].sum(), 1e-9))
    return out


def psnr(a, b):
    mse = np.mean((a-b)**2)
    return float(10*np.log10(255.0**2/mse)) if mse > 0 else 99.0


def fingerprint(deg, gt):
    s_d, s_g = stats(deg), stats(gt)
    br = band_ratio(deg, gt)
    return dict(grad_ratio=s_d["grad"]/max(s_g["grad"],1e-6),
                lapvar_ratio=s_d["lapvar"]/max(s_g["lapvar"],1e-6),
                noise_ratio=s_d["noise"]/max(s_g["noise"],1e-6),
                psnr=psnr(deg, gt), **br)


# ============ 退化原子操作 ============
def jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(np.clip(img,0,255).astype(np.uint8)).save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)


def gauss(img, s):
    return cv2.GaussianBlur(img, (0,0), s) if s > 0 else img


def bilat(img, d, sc, ss):
    return cv2.bilateralFilter(np.clip(img,0,255).astype(np.uint8), d, sc, ss).astype(np.float32)


def build_chain(sigma, q, use_bilat, bilat_s):
    """退化链：可选双边 → 高斯低通 → JPEG"""
    def f(gt):
        x = gt
        if use_bilat:
            x = bilat(x, 9, bilat_s, bilat_s)
        x = gauss(x, sigma)
        if q < 100:
            x = jpeg(x, q)
        return x
    return f


# ============ 采样真实指纹 ============
print("=" * 96)
print("步骤 1：真实 LQ 指纹（分图统计，因三图退化强度差异大）")
print("=" * 96)

per_image_fp, crop_store = {}, []
for lqn, gtn in PAIRS:
    lq_full = load(os.path.join(D, lqn)); gt_full = load(os.path.join(D, gtn))
    if lq_full.shape != gt_full.shape:
        continue
    rng = np.random.RandomState(abs(hash(lqn)) % 1000)
    h, w, _ = lq_full.shape
    fps = []
    for _ in range(8):
        y, x = rng.randint(0, h-512), rng.randint(0, w-512)
        gt, lq = gt_full[y:y+512, x:x+512], lq_full[y:y+512, x:x+512]
        if gt.std() < 12:
            continue
        fps.append(fingerprint(lq, gt))
        crop_store.append((gt, lq, lqn))
    if fps:
        per_image_fp[lqn] = {k: float(np.mean([f[k] for f in fps])) for k in fps[0]}

KEYS = ["grad_ratio","lapvar_ratio","noise_ratio","low","mid","high","psnr"]
print(f"{'图':<16}" + "".join(f"{k:>12}" for k in KEYS))
for n, fp in per_image_fp.items():
    print(f"{n:<16}" + "".join(f"{fp[k]:>12.4f}" for k in KEYS))

all_fp = [fingerprint(lq, gt) for gt, lq, _ in crop_store]
real = {k: float(np.mean([f[k] for f in all_fp])) for k in KEYS}
real_sd = {k: float(np.std([f[k] for f in all_fp])) for k in KEYS}
print(f"{'[全局均值]':<16}" + "".join(f"{real[k]:>12.4f}" for k in KEYS))
print(f"{'[标准差]':<16}" + "".join(f"{real_sd[k]:>12.4f}" for k in KEYS))

# ============ 网格搜索 ============
print()
print("=" * 96)
print("步骤 2：退化链网格搜索（双边? → 高斯σ → JPEG q）")
print("=" * 96)

scale = {k: max(real_sd[k], abs(real[k])*0.1, 1e-3) for k in KEYS}
W = dict(grad_ratio=1.0, lapvar_ratio=2.0, noise_ratio=1.5,
         low=2.0, mid=2.0, high=2.0, psnr=1.5)

eval_crops = crop_store[:15]
grid = list(itertools.product(
    [0.6, 0.8, 1.0, 1.2, 1.4],          # gauss sigma
    [95, 85, 75, 60, 100],               # jpeg quality (100 = 不压)
    [(False, 0), (True, 50), (True, 90)] # (用双边, 强度)
))

results = []
for sigma, q, (ub, bs) in grid:
    fn = build_chain(sigma, q, ub, bs)
    fps = [fingerprint(fn(gt), gt) for gt, _, _ in eval_crops]
    m = {k: float(np.mean([f[k] for f in fps])) for k in KEYS}
    dist = sum(W[k]*abs(m[k]-real[k])/scale[k] for k in KEYS) / sum(W.values())
    tag = f"g{sigma}_q{q}" + (f"_bil{bs}" if ub else "")
    results.append((dist, tag, m))

results.sort()
hdr = f"{'退化链':<20}{'距离':>7}" + "".join(f"{k[:9]:>10}" for k in KEYS)
print(hdr)
print(f"{'[真实 LQ]':<20}{'--':>7}" + "".join(f"{real[k]:>10.3f}" for k in KEYS))
print("-" * len(hdr))
for dist, tag, m in results[:14]:
    print(f"{tag:<20}{dist:>7.3f}" + "".join(f"{m[k]:>10.3f}" for k in KEYS))

best = results[0]
print()
print(f"最优退化链：{best[1]}   距离 {best[0]:.3f}")

json.dump({"real": real, "real_sd": real_sd, "per_image": per_image_fp,
           "grid": [{"tag": t, "dist": d, "stats": m} for d, t, m in results]},
          open(os.path.join(OUT, "calib_v2.json"), "w"), indent=1, ensure_ascii=False)
print(f"结果已存 {OUT}/calib_v2.json")
