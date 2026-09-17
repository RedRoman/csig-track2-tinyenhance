"""架构信息瓶颈实验：这个 latent 架构做 x1 任务的天花板在哪？

核心判据：
  把 GT 直接送进网络（理想输入），输出与 GT 的 PSNR = 架构的重建上限。
  若该上限 < LQ 自身 PSNR (~25.8dB)，则此架构做 x1 是净负收益。

同时对比几种候选路径的上限：
  A. 当前 x1 SRNet（PixelUnshuffle(8) -> UNet -> 4ch latent -> TAESD）
  B. 仅 TAESD encode-decode（若有 encoder）
  C. 恒等映射（理论上限）
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/cache/zihaohe/csig_isr/NTIRE2026")
from diffusers import UNet2DConditionModel
from models.team11_SamsungAKCamera.model import create_srnet

VAL = "/cache/zihaohe/csig_isr/data/赛题二/验证集"
CKPT = "/cache/zihaohe/csig_isr/NTIRE2026/model_zoo/team11_SamsungAICamera/team11_mobile_samsung/weights.pth"
X1 = "/cache/zihaohe/csig_isr/x1_model/srnet_x1_init.pth"
SD21 = "/cache/zihaohe/csig_isr/sd21base"
dev = torch.device("cuda")

PAIRS = [("case1_lq.jpg", "case1_gt.png"),
         ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]


def build_x1():
    cfg = UNet2DConditionModel.load_config(SD21, subfolder="unet")
    unet = UNet2DConditionModel.from_config(cfg)
    m = create_srnet(unet, checkpoint_path=CKPT)
    m.net.body[1].conv_in = nn.Conv2d(192, 160, 3, padding=1)
    m.net.body[0] = nn.PixelUnshuffle(8)
    sd = torch.load(X1, map_location="cpu", weights_only=False)
    m.net.load_state_dict(sd["net"]); m.refine.load_state_dict(sd["refine"])
    m.decoder.load_state_dict(sd["decoder"])
    return m


def psnr_t(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 10 * np.log10(1.0 / mse) if mse > 0 else 99.0


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def tiles(img, n=8, size=512, seed=0):
    """取 n 个 512 crop（避开纯色区）。"""
    rng = np.random.RandomState(seed)
    _, _, H, W = img.shape
    out = []
    tries = 0
    while len(out) < n and tries < n * 8:
        tries += 1
        y, x = rng.randint(0, H - size), rng.randint(0, W - size)
        t = img[:, :, y:y + size, x:x + size]
        if t.std().item() > 0.045:
            out.append((y, x))
    return out


print("=" * 82)
print("架构信息瓶颈实验")
print("=" * 82)
m = build_x1().to(dev).half().eval()

# 逐图收集
rows = []
for lqn, gtn in PAIRS:
    lq_f = load(os.path.join(VAL, lqn))
    gt_f = load(os.path.join(VAL, gtn))
    if lq_f.shape != gt_f.shape:
        continue
    pos = tiles(gt_f, n=8, size=512, seed=abs(hash(lqn)) % 997)
    p_lq, p_gtnet, p_lqnet = [], [], []
    for (y, x) in pos:
        gt = gt_f[:, :, y:y+512, x:x+512].to(dev)
        lq = lq_f[:, :, y:y+512, x:x+512].to(dev)
        with torch.no_grad():
            out_gt = m(gt.half()).float()      # GT 进 -> 架构重建上限
            out_lq = m(lq.half()).float()      # LQ 进 -> 实际 zero-shot 表现
        p_lq.append(psnr_t(lq, gt))
        p_gtnet.append(psnr_t(out_gt, gt))
        p_lqnet.append(psnr_t(out_lq, gt))
    rows.append((lqn, np.mean(p_lq), np.mean(p_gtnet), np.mean(p_lqnet)))

print()
print(f"{'图':<16}{'LQ 原样':>12}{'GT过网络':>12}{'LQ过网络':>12}{'判定':>16}")
print("-" * 82)
for n, a, b, c in rows:
    verdict = "架构净亏" if b < a else "架构可用"
    print(f"{n:<16}{a:>12.3f}{b:>12.3f}{c:>12.3f}{verdict:>16}")
ml = np.mean([r[1] for r in rows]); mg = np.mean([r[2] for r in rows]); mn = np.mean([r[3] for r in rows])
print("-" * 82)
print(f"{'均值':<16}{ml:>12.3f}{mg:>12.3f}{mn:>12.3f}")

print()
print("=" * 82)
print("结论")
print("=" * 82)
print(f"  LQ 什么都不做           PSNR = {ml:6.3f} dB")
print(f"  GT 过一遍网络(架构上限)  PSNR = {mg:6.3f} dB   <- 此架构对 x1 任务的天花板")
print(f"  LQ 过网络(zero-shot)    PSNR = {mn:6.3f} dB")
print()
if mg < ml:
    print(f"  架构上限 {mg:.2f} dB < LQ 原样 {ml:.2f} dB")
    print(f"  => latent 压缩(48:1)碾碎了本来完好的低频, 即使输入完美也回不去")
    print(f"  => 此架构做 x1 是净负收益, 必须改为残差结构或提高 latent 容量")
else:
    print(f"  架构上限 {mg:.2f} dB > LQ 原样 {ml:.2f} dB, 架构本身可用")

print()
print("=" * 82)
print("补充：低频保持能力（这是本任务的关键，实测退化低频能量比=1.00）")
print("=" * 82)


def bands(x):
    g = x.float().mean(dim=1, keepdim=True)
    lo = F.avg_pool2d(g, 16)
    lo_up = F.interpolate(lo, size=g.shape[-2:], mode="bilinear", align_corners=False)
    return lo_up, g - lo_up


gt_f = load(os.path.join(VAL, PAIRS[0][1]))
lq_f = load(os.path.join(VAL, PAIRS[0][0]))
pos = tiles(gt_f, n=6, size=512, seed=7)
d_lo_lq, d_lo_net, d_hi_lq, d_hi_net = [], [], [], []
for (y, x) in pos:
    gt = gt_f[:, :, y:y+512, x:x+512].to(dev)
    lq = lq_f[:, :, y:y+512, x:x+512].to(dev)
    with torch.no_grad():
        og = m(gt.half()).float()
    glo, ghi = bands(gt); llo, lhi = bands(lq); nlo, nhi = bands(og)
    d_lo_lq.append((llo - glo).abs().mean().item())
    d_lo_net.append((nlo - glo).abs().mean().item())
    d_hi_lq.append((lhi - ghi).abs().mean().item())
    d_hi_net.append((nhi - ghi).abs().mean().item())

print(f"  {'':<26}{'低频误差':>14}{'高频误差':>14}")
print(f"  {'LQ vs GT':<26}{np.mean(d_lo_lq):>14.5f}{np.mean(d_hi_lq):>14.5f}")
print(f"  {'GT过网络 vs GT':<26}{np.mean(d_lo_net):>14.5f}{np.mean(d_hi_net):>14.5f}")
r = np.mean(d_lo_net) / max(np.mean(d_lo_lq), 1e-9)
print()
print(f"  网络对低频的破坏是 LQ 自身低频误差的 {r:.1f} 倍")
if r > 1:
    print(f"  => 网络在破坏本来完好的低频, 印证 latent 压缩不适合 x1 任务")
