"""退化强度校准探针：合成训练对 vs 真实验证对的感知距离对齐。

问题：训练域 LPIPS(LQ,GT)≈0.2-0.28（势垒 0.30 从未咬合），真实验证域 0.4262。
合成退化太轻 → (a) 模型在简单模式下训练；(b) 一切 FR 势垒在真实域上形同虚设；
(c) 教师一放开 IQA 就在真实域崩 FR（teacher_v2 刚发生）。

测：现行 legacy 链、以及三档加强链在 512 crop 上的 LPIPS/DISTS 分布，
找到与真实验证域（LPIPS 0.4262 / DISTS 0.2942）对齐的档位。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, cv2, random
from PIL import Image
from settings import ROOT
from degradation import degrade as degrade_legacy
from metrics_utils import create_metrics

Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")
M = create_metrics(DEV)
N = 60


def stronger(gt, rng, blur_sigma, jpeg_q, bil_sigma):
    """加强版链：双边(σ可调) → 高斯(σ可调) → JPEG(q可调)，结构同 legacy"""
    img = gt.copy()
    img = cv2.bilateralFilter(img, d=9, sigmaColor=bil_sigma, sigmaSpace=bil_sigma)
    k = int(blur_sigma * 4) | 1
    img = cv2.GaussianBlur(img, (k, k), blur_sigma)
    enc = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])[1]
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
    return np.ascontiguousarray(img)


paths = sorted(__import__("glob").glob(os.path.join(ROOT, "train_data", "lsdir512", "*.png")))
rng0 = np.random.RandomState(7)
pick = rng0.choice(len(paths), N, replace=False)

CHAINS = {
    "legacy(现行)": None,
    "A: σ2.0/q50/bil120": (2.0, 50, 120),
    "B: σ2.6/q40/bil150": (2.6, 40, 150),
    "C: σ3.2/q30/bil180": (3.2, 30, 180),
}
to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.).permute(2, 0, 1)[None].to(DEV)
print(f"{'链':<22}{'LPIPS':>9}{'DISTS':>9}")
for tag, cfg in CHAINS.items():
    torch.manual_seed(1); np.random.seed(1); random.seed(1)
    ls, ds = [], []
    for i in pick:
        rng = np.random.RandomState(int(i))
        gt = np.asarray(Image.open(paths[i]).convert("RGB"), np.uint8)
        if gt.shape[0] != 512 or gt.shape[1] != 512:
            gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_CUBIC)
        lq = degrade_legacy(gt, rng) if cfg is None else stronger(gt, rng, *cfg)
        with torch.no_grad():
            ls.append(float(M["lpips"](to_t(lq), to_t(gt))))
            ds.append(float(M["dists"](to_t(lq), to_t(gt))))
    print(f"{tag:<22}{np.mean(ls):>9.4f}{np.mean(ds):>9.4f}", flush=True)
print(f"{'真实验证域(目标)':<22}{'0.4262':>9}{'0.2942':>9}")
