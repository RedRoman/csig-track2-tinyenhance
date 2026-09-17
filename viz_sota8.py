"""8 张 GT 的三列对比图：标定退化 LQ | SOTA(distill_v2b, 榜 85.2957) | GT。

GT 从 DIV2K-Val HR patch 里按内容多样性挑 8 张，
退化 = 训练用的标定链 degrade()（双边+高斯+JPEG，随机档），
每行同一确定性种子，可复现。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, cv2
from PIL import Image, ImageDraw
from settings import ROOT
from degradation import degrade as degrade_legacy
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
CK = f"{R}/train_out/distill_v2b/best.pth"
S = 512

import glob
picks = sorted(glob.glob("/cache/zihaohe/gt8/*.png"))
print(f"选中 {len(picks)} 张:", [os.path.basename(p) for p in picks], flush=True)

model, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=False,
                       alpha_mode="fixed", fixed_alpha=1.0)
model.eval()

rows = []
for idx, p in enumerate(picks):
    gt = np.asarray(Image.open(p).convert("RGB"), np.uint8)
    h, w, _ = gt.shape
    if h < S or w < S:
        gt = cv2.resize(gt, (max(S, w), max(S, h)), interpolation=cv2.INTER_CUBIC)
        h, w, _ = gt.shape
    y = (h - S) // 2; x = (w - S) // 2
    gt = np.ascontiguousarray(gt[y:y + S, x:x + S])
    rng = np.random.RandomState(1000 + idx * 37)
    lq = degrade_legacy(gt, rng)
    t = torch.from_numpy(lq.astype(np.float32) / 255.).permute(2, 0, 1)[None].to(DEV, dtype=DT)
    with torch.no_grad(), torch.autocast("cuda", dtype=DT):
        sr = model(t).float().clamp(0, 1)
    sr = (sr[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    mse = float(np.mean((lq.astype(np.float32) - gt.astype(np.float32)) ** 2))
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-6))
    rows.append((os.path.basename(p), psnr, lq, sr, gt))
    print(f"  行{idx+1} {os.path.basename(p)}  LQ_vs_GT {psnr:.1f} dB", flush=True)

pad, hdr, lab = 6, 30, 150
cols = ["LQ (标定退化)", "SOTA distill_v2b (榜85.30)", "GT"]
Wt = lab + 3 * S + 4 * pad
Ht = hdr + len(rows) * (S + pad) + pad
canvas = Image.new("RGB", (Wt, Ht), (20, 20, 20))
d = ImageDraw.Draw(canvas)
for c, name in enumerate(cols):
    d.text((lab + pad + c * (S + pad) + 6, 8), name, fill=(240, 240, 240))
for r, (name, psnr, lq, sr, gt) in enumerate(rows):
    y0 = hdr + r * (S + pad)
    d.text((6, y0 + S // 2 - 12), f"{name[:14]}", fill=(200, 200, 200))
    d.text((6, y0 + S // 2 + 6), f"LQ {psnr:.1f}dB", fill=(150, 150, 150))
    for c, img in enumerate((lq, sr, gt)):
        canvas.paste(Image.fromarray(img), (lab + pad + c * (S + pad), y0))
out = "/cache/zihaohe/viz_sota8.png"
canvas.save(out, quality=95)
print("已存", out, canvas.size, flush=True)
