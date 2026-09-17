"""教师同场对比：PixelDiT@240k vs teacher_v3_mixlq（VOSR），同 200 张确定性 mix-LQ。

判据：FR（LPIPS/DISTS/PSNR，教师的核心价值）+ lf_top（幻觉）。
NR 不看——教师的 NR 由学生侧 IQA 损失补，教师只要保真和干净。
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from metrics_utils import create_metrics

DEV = torch.device("cuda")
M = create_metrics(DEV)
D = "/cache/zihaohe/distill_data"

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV)

sr_map = {}
for f in glob.glob(f"{D}/pdit_pilot_sr/*_sr.png"):
    idx = os.path.basename(f).split("_")[1]
    sr_map[idx] = f

def stat(rows, tag):
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"{tag:<24} lpips={m['lpips']:.4f} dists={m['dists']:.4f} "
          f"psnr={m['psnr']:.2f} lf_top={m['lf_top']:.4f}", flush=True)

rows_p, rows_v, rows_l, rows_2 = [], [], [], []
strong_p, strong_v, strong_2 = [], [], []
import cv2, io
def crop_seed(i): return (i * 104729 + 7919) % (2 ** 31)
paths = sorted(glob.glob("/cache/zihaohe/csig_isr/train_data/lsdir512/*.png"))[:200]

for i in range(200):
    key = f"{i:06d}"
    gt = load(f"{D}/pilot_gt/{key}.png")
    lq = load(f"{D}/lq_mix05/{key}.png")
    # 判定该 idx 是否强档（复现 rng）
    rng = np.random.RandomState(crop_seed(i))
    im = Image.open(paths[i]); h, w = im.size[1], im.size[0]
    if h > 512 or w > 512:
        rng.randint(0, max(h-512,0)+1); rng.randint(0, max(w-512,0)+1)
    rng.random(); rng.randint(4)
    is_strong = rng.random() < 0.5
    for src, rows, srows in ((f"{D}/pdit_pilot_sr/{sr_map[key].split('/')[-1]}", rows_p, strong_p),
                             (f"{D}/teacher_v3_mixlq/{key}.png", rows_v, strong_v),
                             (f"{D}/vosr2_pilot_sr/{key}.png", rows_2, strong_2),
                             (None, rows_l, None)):
        x = lq if src is None else load(src)
        r = {"lpips": float(M["lpips"](x, gt)), "dists": float(M["dists"](x, gt)),
             "psnr": 10 * np.log10(1.0 / max(float(((x - gt) ** 2).mean()), 1e-12))}
        lo = F.avg_pool2d(x, 16); ll = F.avg_pool2d(lq, 16)
        dev_ = (lo - ll).abs().mean(1).flatten()
        r["lf_top"] = float(torch.topk(dev_, max(1, int(dev_.numel() * 0.001))).values.mean())
        rows.append(r)
        if srows is not None and is_strong:
            srows.append(r)

stat(rows_l, "LQ 基线")
stat(rows_v, "VOSR teacher_v3_mixlq")
stat(rows_p, "PixelDiT@240k")
stat(rows_2, "VOSR2@1.4B onestep")
print(f"\n强档子集（{len(strong_v)} 张）:")
stat(strong_v, "  VOSR")
stat(strong_p, "  PixelDiT")
stat(strong_2, "  VOSR2")
