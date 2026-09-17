"""离线生成教师输出，供跨架构蒸馏使用。

教师 = 我们微调的 VOSR 0.5B（vosr_cap，全尺度 Score 4.1093，86.51ms）
学生 = TinySR（12.75ms）

数据集用确定性种子（seed = idx*104729 % 2^31），生成脚本和训练脚本共用同一套，
所以只存教师输出即可，lq/gt 两边各自重算，省 2/3 磁盘。
"""
import os, sys, glob, argparse, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--out", default="/cache/zihaohe/distill_data/v8_teacher")
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--mix", type=float, default=0.0)
ap.add_argument("--export_lq", action="store_true", help="只导出确定性 LQ png，不载教师")
args = ap.parse_args()

sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import cv2
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from settings import ROOT
from degradation import degrade as degrade_legacy

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16


def crop_seed(idx):
    """确定性种子——生成端与训练端必须完全一致，否则教师输出对不上 lq"""
    return (idx * 104729 + 7919) % (2 ** 31)


class DetPairDataset(Dataset):
    """与 train_tinysr 的 PairDataset 同样的退化链，但种子只由 idx 决定"""

    def __init__(self, paths, size=512, mix=0.0):
        self.paths = paths
        self.size = int(size)
        self.mix = float(mix)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.RandomState(crop_seed(i))
        gt = np.asarray(Image.open(self.paths[i]).convert("RGB"), dtype=np.uint8)
        h, w, _ = gt.shape
        s = self.size
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
        gt = np.ascontiguousarray(gt)
        # mix 退化（与 train_distill.DistillDataset 逐字一致，rng 采样顺序不可改）
        if self.mix > 0 and rng.random() < self.mix:
            img = cv2.bilateralFilter(gt, d=9, sigmaColor=110, sigmaSpace=110)
            img = cv2.GaussianBlur(img, (9, 9), 1.8)
            import io as _io
            buf = _io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=55)
            buf.seek(0)
            lq = np.ascontiguousarray(np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8))
        else:
            lq = degrade_legacy(gt, rng)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.).permute(2, 0, 1)
        return to_t(lq), to_t(gt), i


def load_teacher(ck_path):
    for m in list(sys.modules):
        if m == "models" or m.startswith("models."):
            del sys.modules[m]
    sys.path.insert(0, os.environ.get("TINYSR_ROOT", "/cache/zihaohe/csig_isr/TinySR"))
    from tinysr_runtime import build_model
    m, _ = build_model(ck_path, device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
    m.eval()
    return m


os.makedirs(args.out, exist_ok=True)
paths = sorted(glob.glob(os.path.join(ROOT, "train_data", "lsdir512", "*.png")))[:args.n]
print(f"源图 {len(paths)} 张，输出 {args.out}", flush=True)
ds = DetPairDataset(paths, args.size, mix=args.mix)
dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

if args.export_lq:
    import time; t0 = time.time(); done = 0
    for lq, gt, idx in dl:
        arr = (lq.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        for j in range(len(idx)):
            p = os.path.join(args.out, f"{int(idx[j]):06d}.png")
            if not os.path.exists(p):
                Image.fromarray(arr[j]).save(p, compress_level=1)
        done += len(idx)
        if done % 2000 < args.bs:
            print(f"  LQ {done}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    print(f"LQ 导出完成 {done} 张，{(time.time()-t0)/60:.1f} 分钟", flush=True)
    sys.exit(0)

model = load_teacher(args.teacher)
done = 0
import time; t0 = time.time()
with torch.no_grad():
    for lq, gt, idx in dl:
        # 已存在则跳过，支持断点续跑
        need = [j for j, i in enumerate(idx.tolist())
                if not os.path.exists(os.path.join(args.out, f"{i:06d}.png"))]
        if not need:
            done += len(idx); continue
        with torch.autocast("cuda", dtype=DT):
            out = model(lq.to(DEV, dtype=DT)).float().clamp(0, 1)
        arr = (out.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        for j in need:
            Image.fromarray(arr[j]).save(os.path.join(args.out, f"{int(idx[j]):06d}.png"),
                                         compress_level=1)
        done += len(idx)
        if done % 400 < args.bs:
            el = time.time() - t0
            print(f"  {done}/{len(paths)}  {el:.0f}s  预计总计 {el/max(done,1)*len(paths)/60:.1f} 分钟",
                  flush=True)
print(f"完成 {done} 张，用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
