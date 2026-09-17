"""并排出图：LQ / vosr_cap / teacher2 / GT，看 MANIQA=1.10、PSNR=18 到底长什么样。

指标能被攻击，眼睛不能。这个项目已经有六次「拿窄观测面下结论」的记录，
其中包括只看指标不看图。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch
from PIL import Image, ImageDraw
from settings import VAL
from vosr_load import load_vosr

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
# 取三个不同退化档的样本位置
PICKS = [("case1_lq.jpg", "case1_gt.png", 0.50, 0.42),
         ("case3_lq.jpg", "case3_gt.jpg", 0.38, 0.55)]
S = 448
MODELS = [("vosr_cap", f"{R}/train_out/vosr_cap/best.pth"),
          ("teacher2", f"{R}/train_out/vosr_teacher2/best.pth")]


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def crop(t, fy, fx, s=S):
    H, W = t.shape[-2:]
    y = int(np.clip(fy * H - s // 2, 0, H - s)); x = int(np.clip(fx * W - s // 2, 0, W - s))
    return t[:, :, y:y + s, x:x + s].contiguous()


rows = []
for lqn, gtn, fy, fx in PICKS:
    lq_full = load(os.path.join(VAL, lqn)); gt_full = load(os.path.join(VAL, gtn))
    lqc = crop(lq_full, fy, fx); gtc = crop(gt_full, fy, fx)
    tiles = [("LQ 输入", lqc[0]), ]
    for tag, ck in MODELS:
        m = load_vosr(ck, DEV, DT, verbose=False)
        # 模型是 512 原生，用 512 块跑再裁中间 448，避免边界效应
        H, W = lq_full.shape[-2:]
        y = int(np.clip(fy * H - 256, 0, H - 512)); x = int(np.clip(fx * W - 256, 0, W - 512))
        blk = lq_full[:, :, y:y + 512, x:x + 512].to(DEV, dtype=DT)
        with torch.no_grad(), torch.autocast("cuda", dtype=DT):
            o = m(blk).float().clamp(0, 1).cpu()
        off = (512 - S) // 2
        tiles.append((tag, o[0, :, off:off + S, off:off + S]))
        del m; torch.cuda.empty_cache()
    tiles.append(("GT 真值", gtc[0]))
    rows.append(tiles)

pad, hdr = 6, 26
cols = len(rows[0])
Wt = cols * S + (cols + 1) * pad
Ht = len(rows) * (S + hdr) + (len(rows) + 1) * pad
canvas = Image.new("RGB", (Wt, Ht), (24, 24, 24))
d = ImageDraw.Draw(canvas)
for r, tiles in enumerate(rows):
    for c, (tag, t) in enumerate(tiles):
        im = Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        x0 = pad + c * (S + pad); y0 = pad + r * (S + hdr + pad)
        d.text((x0 + 4, y0 + 4), tag, fill=(240, 240, 240))
        canvas.paste(im, (x0, y0 + hdr))
out = "/cache/zihaohe/montage_teacher.png"
canvas.save(out, quality=95)
print("已存", out, canvas.size, flush=True)
