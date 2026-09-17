"""导出蒸馏集 idx 0..199 的确定性 GT crop（与 gen_teacher.DetPairDataset GT 侧逐字一致）。"""
import os, sys, glob
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
from settings import ROOT

def crop_seed(idx):
    return (idx * 104729 + 7919) % (2 ** 31)

OUT = "/cache/zihaohe/distill_data/gt_full"
os.makedirs(OUT, exist_ok=True)
paths = sorted(glob.glob(os.path.join(ROOT, "train_data", "lsdir512", "*.png")))[:20000]
s = 512
for i, p in enumerate(paths):
    rng = np.random.RandomState(crop_seed(i))
    gt = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    h, w, _ = gt.shape
    if h > s or w > s:
        y = rng.randint(0, max(h - s, 0) + 1); x = rng.randint(0, max(w - s, 0) + 1)
        gt = gt[y:y + s, x:x + s]
    if gt.shape[0] != s or gt.shape[1] != s:
        gt = cv2.resize(gt, (s, s), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.5:
        gt = gt[:, ::-1]
    k = int(rng.randint(4))
    if k:
        gt = np.rot90(gt, k)
    Image.fromarray(np.ascontiguousarray(gt)).save(f"{OUT}/{i:06d}.png", compress_level=1)
print("GT 导出 20000 张 ->", OUT)
