"""出一批可视化块。两族模型的 models 包会撞名，所以分两趟跑，块存盘后再拼。

用法: viz_tiles.py --family tinysr|vosr|src
"""
import os, sys, argparse, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["tinysr", "vosr", "src"], required=True)
ap.add_argument("--dir", default="/cache/zihaohe/viz_tiles")
args = ap.parse_args()

sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
if args.family == "vosr":
    sys.path.insert(0, "/cache/zihaohe/VOSR")
sys.path.insert(0, "/cache/zihaohe")

import numpy as np, torch
from PIL import Image
from settings import VAL

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
S = 384
# (标签, lq文件, gt文件, 纵向比例, 横向比例)  —— 覆盖重/中/轻三档退化
PICKS = [
    ("A_楼顶", "case1_lq.jpg", "case1_gt.png", 0.50, 0.42),
    ("B_楼面", "case3_lq.jpg", "case3_gt.jpg", 0.38, 0.55),
    ("C_树冠", "case1_lq.jpg", "case1_gt.png", 0.22, 0.60),
    ("D_轻退化", "case2_lq.jpg", "case2_gt.png", 0.55, 0.45),
]
TINYSR = [("run12_基线", f"{R}/train_out/run12_iqa30/best.pth"),
          ("distill_v1_学生", f"{R}/train_out/distill_v1/best.pth")]
VOSR = [("vosr_cap_教师", f"{R}/train_out/vosr_cap/best.pth"),
        ("teacher2_加强版", f"{R}/train_out/vosr_teacher2/best.pth")]

os.makedirs(args.dir, exist_ok=True)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def save(t, name):
    a = (t.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(a).save(os.path.join(args.dir, name + ".png"))


def block_and_center(full, fy, fx):
    """按 512 原生块跑模型，再裁中心 S，避开边界效应"""
    H, W = full.shape[-2:]
    y = int(np.clip(fy * H - 256, 0, H - 512)); x = int(np.clip(fx * W - 256, 0, W - 512))
    return full[:, :, y:y + 512, x:x + 512], (512 - S) // 2


if args.family == "src":
    for tag, lqn, gtn, fy, fx in PICKS:
        lq = load(os.path.join(VAL, lqn)); gt = load(os.path.join(VAL, gtn))
        for nm, full in (("00_LQ输入", lq), ("99_GT真值", gt)):
            blk, off = block_and_center(full, fy, fx)
            save(blk[0, :, off:off + S, off:off + S], f"{tag}__{nm}")
        print(f"  {tag} 源图已存", flush=True)
    sys.exit(0)

cands = TINYSR if args.family == "tinysr" else VOSR
if args.family == "tinysr":
    from tinysr_runtime import build_model

    def get(ck):
        m, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=1.0)
        return m.eval()
else:
    from vosr_load import load_vosr

    def get(ck):
        return load_vosr(ck, DEV, DT, verbose=False)

for i, (tag_m, ck) in enumerate(cands):
    if not os.path.exists(ck):
        print(f"  缺 {tag_m}", flush=True); continue
    m = get(ck)
    idx = (1 if args.family == "tinysr" else 3) + i   # 控制列顺序
    for tag, lqn, gtn, fy, fx in PICKS:
        lq = load(os.path.join(VAL, lqn))
        blk, off = block_and_center(lq, fy, fx)
        with torch.no_grad(), torch.autocast("cuda", dtype=DT):
            o = m(blk.to(DEV, dtype=DT)).float().clamp(0, 1)
        save(o[0, :, off:off + S, off:off + S], f"{tag}__{idx:02d}_{tag_m}")
    print(f"  {tag_m} 完成", flush=True)
    del m; torch.cuda.empty_cache()
