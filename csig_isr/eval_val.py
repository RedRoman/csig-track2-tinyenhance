"""在赛题验证集上评测 x1 模型（zero-shot 基线）。

- 4K 图分块推理（512 tile，overlap 32，高斯权重融合）
- 用 NTIRE 官方口径算 Score:
    Score = (1-LPIPS)+(1-DISTS)+CLIPIQA+MANIQA+MUSIQ/100+max(0,(10-NIQE)/10)
- 同时给出 LQ 自身的 Score 作为下界参照（不做任何处理直接提交能得多少）
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/cache/zihaohe/csig_isr/NTIRE2026")

VAL = "/cache/zihaohe/csig_isr/data/赛题二/验证集"
JIT = "/cache/zihaohe/csig_isr/x1_model/srnet_x1.pt"
OUTDIR = "/cache/zihaohe/csig_isr/eval_out/x1_zeroshot"
os.makedirs(OUTDIR, exist_ok=True)
dev = torch.device("cuda")

PAIRS = [("case1_lq.jpg", "case1_gt.png"),
         ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]

TILE, OVERLAP = 512, 32


def gaussian_weight(h, w, device, dtype):
    """tile 融合用的高斯权重，边缘衰减避免接缝。"""
    yy = torch.linspace(-1, 1, h, device=device)
    xx = torch.linspace(-1, 1, w, device=device)
    gy = torch.exp(-(yy ** 2) / (2 * 0.5 ** 2))
    gx = torch.exp(-(xx ** 2) / (2 * 0.5 ** 2))
    return (gy[:, None] * gx[None, :]).to(dtype)[None, None]


@torch.no_grad()
def tiled_infer(model, img, tile=TILE, overlap=OVERLAP):
    """img: [1,3,H,W] in [0,1] fp16 on cuda"""
    _, _, H, W = img.shape
    stride = tile - overlap
    out = torch.zeros_like(img, dtype=torch.float32)
    acc = torch.zeros(1, 1, H, W, device=img.device, dtype=torch.float32)
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    wmap = gaussian_weight(tile, tile, img.device, torch.float32)
    for y in ys:
        for x in xs:
            patch = img[:, :, y:y + tile, x:x + tile]
            o = model(patch).float()
            out[:, :, y:y + tile, x:x + tile] += o * wmap
            acc[:, :, y:y + tile, x:x + tile] += wmap
    return (out / acc.clamp(min=1e-8)).clamp(0, 1)


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


print("=" * 80)
print("1. 载入 jit 模型")
print("=" * 80)
model = torch.jit.load(JIT, map_location=dev)
model.eval()
print(f"   {JIT}")

print()
print("=" * 80)
print("2. 分块推理 3 组验证集（4K）")
print("=" * 80)
sr_paths, gt_paths, lq_paths = [], [], []
for lqn, gtn in PAIRS:
    lq = load_img(os.path.join(VAL, lqn)).to(dev).half()
    t0 = time.time()
    sr = tiled_infer(model, lq)
    torch.cuda.synchronize()
    dt = time.time() - t0
    arr = (sr[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    op = os.path.join(OUTDIR, lqn.replace("_lq.jpg", ".png"))
    Image.fromarray(arr).save(op)
    sr_paths.append(op)
    gt_paths.append(os.path.join(VAL, gtn))
    lq_paths.append(os.path.join(VAL, lqn))
    print(f"   {lqn:<16} {tuple(lq.shape[2:])} -> {arr.shape[:2]}  {dt:.2f}s")

print()
print("=" * 80)
print("3. 指标计算（NTIRE 官方口径）")
print("=" * 80)
import pyiqa

metrics = {}
for name, kw in [("psnr", dict(test_y_channel=True, color_space="ycbcr")),
                 ("ssim", dict(test_y_channel=True, color_space="ycbcr")),
                 ("lpips", {}), ("dists", {}), ("niqe", {}),
                 ("clipiqa", {}), ("musiq", {}), ("maniqa-pipal", {})]:
    metrics[name] = pyiqa.create_metric(name, device=dev, **kw)
print("   指标模型载入完成")


def evaluate(paths, gts, tag):
    res = {k: [] for k in ["psnr", "ssim", "lpips", "dists", "niqe",
                           "clipiqa", "musiq", "maniqa"]}
    for p, g in zip(paths, gts):
        a = load_img(p).to(dev)
        b = load_img(g).to(dev)
        if a.shape != b.shape:
            b = F.interpolate(b, size=a.shape[2:], mode="bicubic", align_corners=False)
        res["psnr"].append(metrics["psnr"](a, b).item())
        res["ssim"].append(metrics["ssim"](a, b).item())
        res["lpips"].append(metrics["lpips"](a, b).item())
        res["dists"].append(metrics["dists"](a, b).item())
        res["niqe"].append(metrics["niqe"](a).item())
        res["clipiqa"].append(metrics["clipiqa"](a).item())
        res["musiq"].append(metrics["musiq"](a).item())
        res["maniqa"].append(metrics["maniqa-pipal"](a).item())
        del a, b
        torch.cuda.empty_cache()
    m = {k: float(np.mean(v)) for k, v in res.items()}
    m["Score"] = ((1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"]
                  + m["maniqa"] + m["musiq"] / 100 + max(0, (10 - m["niqe"]) / 10))
    return m


print()
r_lq = evaluate(lq_paths, gt_paths, "LQ")
r_sr = evaluate(sr_paths, gt_paths, "SR")

print(f"   {'指标':<12}{'LQ 原图':>12}{'x1 模型':>12}{'变化':>12}")
print("   " + "-" * 48)
for k in ["psnr", "ssim", "lpips", "dists", "niqe", "clipiqa", "musiq", "maniqa"]:
    d = r_sr[k] - r_lq[k]
    print(f"   {k:<12}{r_lq[k]:>12.4f}{r_sr[k]:>12.4f}{d:>+12.4f}")
print("   " + "-" * 48)
print(f"   {'Score':<12}{r_lq['Score']:>12.4f}{r_sr['Score']:>12.4f}{r_sr['Score']-r_lq['Score']:>+12.4f}")

print()
print("=" * 80)
print("4. 换算成最终分（假设 CSIG 同 NTIRE 公式）")
print("=" * 80)
print(f"   {'加速比':>8}{'LQ 直交':>14}{'x1 模型':>14}")
for sp in [1, 5, 10, 20, 40]:
    print(f"   {sp:>8}{2**r_lq['Score'] * sp**0.2:>14.2f}{2**r_sr['Score'] * sp**0.2:>14.2f}")

json.dump({"lq": r_lq, "sr": r_sr}, open(os.path.join(OUTDIR, "metrics.json"), "w"),
          indent=1, ensure_ascii=False)
print()
print(f"结果已存 {OUTDIR}/metrics.json")
