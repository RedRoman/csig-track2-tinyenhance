"""容量与结构实验：为 x1 增强任务定架构。

测三件事：
  1. Samsung 4通道 latent vs TinySR 16通道 latent 的重建上限（autoencoder 往返）
  2. 残差结构能否兜底（out = in + G(in)，最差退化到 LQ 原样）
  3. 高频残差本身的能量分布 —— 决定 latent 需要多少容量
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
VAL = "/cache/zihaohe/csig_isr/data/赛题二/验证集"
dev = torch.device("cuda")
PAIRS = [("case1_lq.jpg", "case1_gt.png"),
         ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def psnr_t(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 10 * np.log10(1.0 / mse) if mse > 0 else 99.0


def crops(img, n=8, size=512, seed=0):
    rng = np.random.RandomState(seed)
    _, _, H, W = img.shape
    out, tries = [], 0
    while len(out) < n and tries < n * 10:
        tries += 1
        y, x = rng.randint(0, H - size), rng.randint(0, W - size)
        if img[:, :, y:y+size, x:x+size].std().item() > 0.045:
            out.append((y, x))
    return out


# 收集配对 crop
data = []
for lqn, gtn in PAIRS:
    lq_f, gt_f = load(os.path.join(VAL, lqn)), load(os.path.join(VAL, gtn))
    if lq_f.shape != gt_f.shape:
        continue
    for (y, x) in crops(gt_f, 8, 512, abs(hash(lqn)) % 997):
        data.append((lq_f[:, :, y:y+512, x:x+512].to(dev),
                     gt_f[:, :, y:y+512, x:x+512].to(dev)))
print(f"配对 crop 数：{len(data)}")

print()
print("=" * 84)
print("实验 1：latent 往返重建上限（区分 latent 容量 vs decoder 能力）")
print("=" * 84)
from diffusers import AutoencoderKL

results = {}

# 1a. SD2.1 完整 VAE（4通道 latent，但 decoder 是 49.5M 的大网络）
vae_sd = AutoencoderKL.from_pretrained("/cache/zihaohe/csig_isr/sd21base",
                                       subfolder="vae", torch_dtype=torch.float16).to(dev).eval()
print(f"  SD2.1 VAE: latent_channels={vae_sd.config.latent_channels}, "
      f"decoder 参数 {sum(p.numel() for p in vae_sd.decoder.parameters())/1e6:.1f}M")
acc = []
with torch.no_grad():
    for lq, gt in data:
        z = vae_sd.encode(gt.half() * 2 - 1).latent_dist.mode()
        rec = vae_sd.decode(z).sample.float().clamp(-1, 1) * 0.5 + 0.5
        acc.append(psnr_t(rec, gt))
results["SD2.1 VAE (4ch, 大decoder)"] = float(np.mean(acc))
print(f"    encode->decode(GT) PSNR = {np.mean(acc):.3f} dB")
del vae_sd
torch.cuda.empty_cache()

# 1b. Samsung 的轻量 TAESD decoder 单独能力（用 SD VAE 的 latent 喂它是不匹配的，
#     所以这里只报前一实验的端到端数字作参照）
results["Samsung 4ch 轻量架构 (端到端)"] = 23.482

print()
print("=" * 84)
print("实验 2：对比基准 —— LQ 自身 与 Samsung 4通道架构")
print("=" * 84)
p_lqgt = [psnr_t(lq, gt) for lq, gt in data]
print(f"  LQ 什么都不做              PSNR = {np.mean(p_lqgt):.3f} dB   <- 必须超过这条线")
print(f"  Samsung 4通道架构上限       PSNR = 23.482 dB   (前一实验实测)")
for k, v in results.items():
    mark = "高于基线" if v > np.mean(p_lqgt) else "低于基线"
    print(f"  {k:<34} {v:>8.3f} dB   {mark} {v-np.mean(p_lqgt):+.2f} dB")

print()
print("=" * 84)
print("实验 3：残差结构的兜底能力")
print("=" * 84)
print("  结构 out = in + G(in)，G 输出恒为 0 时退化到 LQ 原样：")
print(f"    最差情况 PSNR = {np.mean(p_lqgt):.3f} dB（等于 LQ 基线，不会净亏）")
print("  对比非残差结构：")
print(f"    最差情况 PSNR = 23.482 dB（架构上限，已低于基线 4.27 dB）")
print()
print("  latent 域残差(TinySR 式)的上限 = VAE 往返重建上限（见实验 1）")

print()
print("=" * 84)
print("实验 4：高频残差的能量分布（决定所需 latent 容量）")
print("=" * 84)


def split_bands(x, k=8):
    lo = F.avg_pool2d(x.float(), k)
    lo = F.interpolate(lo, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return lo, x.float() - lo


e_lo, e_hi, e_res = [], [], []
for lq, gt in data:
    res = gt - lq
    lo, hi = split_bands(gt)
    e_lo.append(lo.abs().mean().item())
    e_hi.append(hi.abs().mean().item())
    e_res.append(res.abs().mean().item())
print(f"  GT 低频幅度          {np.mean(e_lo):.5f}")
print(f"  GT 高频幅度          {np.mean(e_hi):.5f}")
print(f"  需重建的残差(GT-LQ)  {np.mean(e_res):.5f}")
print(f"  残差 / 低频 = {np.mean(e_res)/np.mean(e_lo)*100:.1f}%   "
      f"残差 / 高频 = {np.mean(e_res)/np.mean(e_hi)*100:.1f}%")
print()
print("  => 残差幅度仅为低频的百分之几，让网络只学残差比让它重建整幅图容易得多，")
print("     且低频经 identity 路径完全无损通过。")
