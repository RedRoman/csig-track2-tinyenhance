"""决定架构：残差模式下，各 decoder 的表达能力上限。

方法：冻结 decoder，用梯度下降优化 latent z，使 decoder(z) 逼近目标。
      得到的 PSNR = 该 decoder 在"网络训练到完美"时的天花板。

对比三种目标：
  A. 目标 = GT 整幅图        (非残差结构，当前做法)
  B. 目标 = GT - LQ 残差     (像素域残差结构)
对比两种 decoder：
  1. Samsung TAESD  1.22M
  2. SD2.1 VAE      49.5M
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/cache/zihaohe/csig_isr/NTIRE2026")
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


def crops(img, n, size, seed):
    rng = np.random.RandomState(seed)
    _, _, H, W = img.shape
    out, tries = [], 0
    while len(out) < n and tries < n * 10:
        tries += 1
        y, x = rng.randint(0, H - size), rng.randint(0, W - size)
        if img[:, :, y:y+size, x:x+size].std().item() > 0.045:
            out.append((y, x))
    return out


data = []
for lqn, gtn in PAIRS:
    lq_f, gt_f = load(os.path.join(VAL, lqn)), load(os.path.join(VAL, gtn))
    if lq_f.shape != gt_f.shape:
        continue
    for (y, x) in crops(gt_f, 4, 512, abs(hash(lqn)) % 997):
        data.append((lq_f[:, :, y:y+512, x:x+512].to(dev),
                     gt_f[:, :, y:y+512, x:x+512].to(dev)))
print(f"配对 crop 数：{len(data)}")

# ---- 载入两种 decoder ----
from diffusers import UNet2DConditionModel, AutoencoderKL
from models.team11_SamsungAKCamera.model import create_srnet

CKPT = "/cache/zihaohe/csig_isr/NTIRE2026/model_zoo/team11_SamsungAICamera/team11_mobile_samsung/weights.pth"
SD21 = "/cache/zihaohe/csig_isr/sd21base"

cfg = UNet2DConditionModel.load_config(SD21, subfolder="unet")
srnet = create_srnet(UNet2DConditionModel.from_config(cfg), checkpoint_path=CKPT)
dec_tae = srnet.decoder.to(dev).float().eval()
for p in dec_tae.parameters():
    p.requires_grad_(False)

vae_sd = AutoencoderKL.from_pretrained(SD21, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
dec_sd = vae_sd.decoder
for p in dec_sd.parameters():
    p.requires_grad_(False)

n_tae = sum(p.numel() for p in dec_tae.parameters())
n_sd = sum(p.numel() for p in dec_sd.parameters())
print(f"Samsung TAESD decoder {n_tae/1e6:.2f}M | SD2.1 VAE decoder {n_sd/1e6:.2f}M")


def fit_latent(decoder, target, ch, steps=400, lr=0.05, post=None):
    """冻结 decoder，优化 latent 逼近 target，返回最优 PSNR。"""
    z = torch.zeros(1, ch, 64, 64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    best = -1e9
    for i in range(steps):
        opt.zero_grad()
        out = decoder(z)
        if post is not None:
            out = post(out)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
        if i % 50 == 0 or i == steps - 1:
            with torch.no_grad():
                o = decoder(z)
                if post is not None:
                    o = post(o)
                best = max(best, psnr_t(o, target))
    return best


print()
print("=" * 88)
print("目标 A：直接表达 GT 整幅图（当前非残差结构）")
print("=" * 88)
a_tae, a_sd = [], []
for lq, gt in data:
    a_tae.append(fit_latent(dec_tae, gt, 4))
    a_sd.append(fit_latent(dec_sd, gt * 2 - 1, 4, post=lambda o: o * 0.5 + 0.5))
print(f"  Samsung TAESD (1.22M)  上限 = {np.mean(a_tae):.3f} dB")
print(f"  SD2.1 VAE     (49.5M)  上限 = {np.mean(a_sd):.3f} dB")

print()
print("=" * 88)
print("目标 B：只表达残差 GT-LQ（像素域残差结构 out = LQ + decoder(z)）")
print("=" * 88)
b_tae, b_sd = [], []
for lq, gt in data:
    res = gt - lq
    # decoder 输出经平移缩放到残差量程
    b_tae.append(fit_latent(dec_tae, res + 0.5, 4, post=lambda o: o) - 0)
    b_sd.append(fit_latent(dec_sd, res * 2, 4, post=lambda o: o * 0.5))
# 换算成最终图像 PSNR：out = LQ + rec_res
fin_tae, fin_sd = [], []
for (lq, gt), pt, ps in zip(data, b_tae, b_sd):
    # 由残差 PSNR 反推：残差误差能量相同 => 最终图像误差能量相同
    fin_tae.append(pt)
    fin_sd.append(ps)
print(f"  Samsung TAESD (1.22M)  残差重建 PSNR = {np.mean(b_tae):.3f} dB")
print(f"  SD2.1 VAE     (49.5M)  残差重建 PSNR = {np.mean(b_sd):.3f} dB")
print("  (残差重建 PSNR 即最终图像 PSNR，因 out=LQ+rec_res 的误差 = 残差重建误差)")

print()
print("=" * 88)
print("汇总（LQ 基线 = 26.93 dB，必须超过）")
print("=" * 88)
base = np.mean([psnr_t(lq, gt) for lq, gt in data])
rows = [
    ("非残差 + Samsung TAESD 1.22M", np.mean(a_tae)),
    ("非残差 + SD2.1 VAE 49.5M", np.mean(a_sd)),
    ("像素残差 + Samsung TAESD 1.22M", np.mean(b_tae)),
    ("像素残差 + SD2.1 VAE 49.5M", np.mean(b_sd)),
]
print(f"  {'结构':<36}{'上限 dB':>10}{'vs 基线':>12}{'判定':>10}")
print("  " + "-" * 68)
print(f"  {'LQ 什么都不做':<36}{base:>10.3f}{0.0:>+12.2f}{'基线':>10}")
for name, v in rows:
    print(f"  {name:<36}{v:>10.3f}{v-base:>+12.2f}{'可用' if v > base else '净亏':>10}")
