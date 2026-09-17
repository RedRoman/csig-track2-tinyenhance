"""100 张输出的自动异常检测 + 可疑样本挑选。

教训：先前只看 2-3 个 crop 就下"视觉验证通过"的结论，漏掉了重度退化的 case3
（全档位都在编造伪纹理）。这次对全部 100 张做定量筛查。

检测项：
  1. 过锐化：输出梯度能量 / LQ 梯度能量 过高
  2. 伪纹理：高频能量比 GT 侧统计异常（无 GT 时用局部方差分布的峰度）
  3. tile 接缝：在 tile 边界处的梯度突变
  4. 色偏：与 LQ 的低频均值差
  5. 过曝/欠曝：饱和像素比例变化
"""
import os, sys, json, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
LQ_DIR = f"{ROOT}/data/赛题二/测试集"
SR_DIR = f"{ROOT}/submit/output_dir"
OUT = f"{ROOT}/visualizations/audit"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
TILE, OVERLAP = 512, 128


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def grad_energy(x):
    g = x.mean(1, keepdim=True)
    gx = g[:, :, :, 1:] - g[:, :, :, :-1]
    gy = g[:, :, 1:, :] - g[:, :, :-1, :]
    return float(torch.sqrt((gx ** 2).mean() + (gy ** 2).mean()))


def band_ratio(x, k=16):
    g = x.mean(1, keepdim=True)
    lo = F.avg_pool2d(g, k)
    lo = F.interpolate(lo, size=g.shape[-2:], mode="bilinear", align_corners=False)
    hi = g - lo
    return float(lo.abs().mean()), float(hi.abs().mean())


def seam_score(x, tile=TILE, overlap=OVERLAP):
    """检测 tile 接缝：比较边界列/行的梯度与邻域梯度的比值。"""
    g = x.mean(1, keepdim=True)
    _, _, H, W = g.shape
    stride = tile - overlap
    gx = (g[:, :, :, 1:] - g[:, :, :, :-1]).abs()[0, 0]
    gy = (g[:, :, 1:, :] - g[:, :, :-1, :]).abs()[0, 0]
    worst = 0.0
    for x0 in range(stride, W - 1, stride):
        if x0 + 3 >= gx.shape[1] or x0 < 3:
            continue
        at = gx[:, x0].mean().item()
        nb = torch.cat([gx[:, x0 - 3:x0 - 1], gx[:, x0 + 2:x0 + 4]], 1).mean().item()
        worst = max(worst, at / max(nb, 1e-8))
    for y0 in range(stride, H - 1, stride):
        if y0 + 3 >= gy.shape[0] or y0 < 3:
            continue
        at = gy[y0, :].mean().item()
        nb = torch.cat([gy[y0 - 3:y0 - 1, :], gy[y0 + 2:y0 + 4, :]], 0).mean().item()
        worst = max(worst, at / max(nb, 1e-8))
    return worst


files = sorted(glob.glob(f"{SR_DIR}/*.jpg"),
               key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
print(f"审计 {len(files)} 张", flush=True)

rows = []
for i, sp in enumerate(files, 1):
    name = os.path.basename(sp)
    lp = os.path.join(LQ_DIR, name)
    if not os.path.exists(lp):
        continue
    lq = load(lp).to(dev)
    sr = load(sp).to(dev)
    if lq.shape != sr.shape:
        print(f"  [{name}] 尺寸不符 {tuple(lq.shape)} vs {tuple(sr.shape)}")
        continue
    ge_l, ge_s = grad_energy(lq), grad_energy(sr)
    lo_l, hi_l = band_ratio(lq)
    lo_s, hi_s = band_ratio(sr)
    sat_l = float((lq > 0.99).float().mean())
    sat_s = float((sr > 0.99).float().mean())
    rows.append({
        "name": name,
        "grad_ratio": ge_s / max(ge_l, 1e-8),
        "hf_ratio": hi_s / max(hi_l, 1e-8),
        "lf_shift": abs(lo_s - lo_l),
        "seam": seam_score(sr),
        "sat_delta": sat_s - sat_l,
    })
    del lq, sr
    torch.cuda.empty_cache()
    if i % 20 == 0:
        print(f"  {i}/{len(files)}", flush=True)

arr = {k: np.array([r[k] for r in rows]) for k in
       ["grad_ratio", "hf_ratio", "lf_shift", "seam", "sat_delta"]}
print()
print("=" * 78)
print("全量统计")
print("=" * 78)
print(f"  {'指标':<14}{'均值':>10}{'中位':>10}{'P95':>10}{'最大':>10}")
for k in arr:
    a = arr[k]
    print(f"  {k:<14}{a.mean():>10.4f}{np.median(a):>10.4f}"
          f"{np.percentile(a,95):>10.4f}{a.max():>10.4f}")

# 阈值：偏离中位数太多的视为可疑
sus = []
for r in rows:
    flags = []
    if r["grad_ratio"] > np.percentile(arr["grad_ratio"], 97):
        flags.append("过锐化")
    if r["hf_ratio"] > np.percentile(arr["hf_ratio"], 97):
        flags.append("高频异常")
    if r["lf_shift"] > np.percentile(arr["lf_shift"], 97):
        flags.append("色偏")
    if r["seam"] > 1.35:
        flags.append("接缝")
    if abs(r["sat_delta"]) > 0.02:
        flags.append("过曝变化")
    if flags:
        sus.append((r["name"], flags, r))

print()
print("=" * 78)
print(f"可疑样本 {len(sus)} / {len(rows)}")
print("=" * 78)
for n, f, r in sus[:25]:
    print(f"  {n:<14}{','.join(f):<26}grad{r['grad_ratio']:.3f} hf{r['hf_ratio']:.3f} "
          f"seam{r['seam']:.3f} sat{r['sat_delta']:+.4f}")

json.dump({"rows": rows, "suspicious": [s[0] for s in sus]},
          open(f"{ROOT}/eval_out/audit100.json", "w"), indent=1, ensure_ascii=False)

# 为可疑样本导出对比图
top = [s[0] for s in sus[:6]]
if top:
    print(f"\n导出可疑样本对比图 {len(top)} 张 ...")
    for n in top:
        lq = Image.open(os.path.join(LQ_DIR, n)).convert("RGB")
        sr = Image.open(os.path.join(SR_DIR, n)).convert("RGB")
        W, H = lq.size
        y, x = H // 2 - 256, W // 2 - 256
        box = (x, y, x + 512, y + 512)
        cv = Image.new("RGB", (1030, 512), (20, 20, 20))
        cv.paste(lq.crop(box), (0, 0))
        cv.paste(sr.crop(box), (518, 0))
        cv.save(f"{OUT}/{n.replace('.jpg','')}_cmp.png")
    print(f"  已存 {OUT}")
print("\n审计完成")
