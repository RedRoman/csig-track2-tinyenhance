"""查 TinySR 这一族（含蒸馏产物）有没有同样的物体幻觉。

teacher2 在楼体上凭空生成整片仙人掌状物体，vosr_cap 程度轻但同病。
distill_v1 是拿 vosr_cap 的输出当目标训的，必须确认它有没有继承。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image, ImageDraw
from settings import VAL
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
PICKS = [("case1_lq.jpg", "case1_gt.png", 0.50, 0.42),
         ("case3_lq.jpg", "case3_gt.jpg", 0.38, 0.55)]
S = 448
MODELS = [("run12", f"{R}/train_out/run12_iqa30/best.pth"),
          ("mosaic_cap", f"{R}/train_out/mosaic_cap/best.pth"),
          ("distill_v1", f"{R}/train_out/distill_v1/best.pth")]


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
    tiles = [("LQ 输入", crop(lq_full, fy, fx)[0])]
    for tag, ck in MODELS:
        if not os.path.exists(ck):
            continue
        m, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=1.0)
        m.eval()
        H, W = lq_full.shape[-2:]
        y = int(np.clip(fy * H - 256, 0, H - 512)); x = int(np.clip(fx * W - 256, 0, W - 512))
        blk = lq_full[:, :, y:y + 512, x:x + 512].to(DEV, dtype=DT)
        with torch.no_grad(), torch.autocast("cuda", dtype=DT):
            o = m(blk).float().clamp(0, 1).cpu()
        off = (512 - S) // 2
        tiles.append((tag, o[0, :, off:off + S, off:off + S]))
        del m; torch.cuda.empty_cache()
    tiles.append(("GT 真值", crop(gt_full, fy, fx)[0]))
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
out = "/cache/zihaohe/montage_tinysr.png"
canvas.save(out, quality=95)
print("已存", out, canvas.size, flush=True)
