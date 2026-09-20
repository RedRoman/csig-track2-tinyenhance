"""渲染测试集若干张的 512 crop 对比条(LQ | 各候选 | 参照),用于肉眼查"眼睛"幻觉。
用法: viz_cands.py OUTDIR ref=ckpt cand1=ckpt cand2=ckpt ...  (第一个为参照放最右)"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch
from PIL import Image, ImageDraw
from pix_model import build_pix
Image.MAX_IMAGE_PIXELS = None
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
specs = [a.split("=", 1) for a in sys.argv[2:]]
TEST = "/cache/zihaohe/csig_isr/data/赛题二/测试集"
# (文件, y0, x0):树冠/墙面/窗户等易长"眼睛"的区域,取 512 crop
CASES = [("case1.jpg", 1200, 1800), ("case1.jpg", 400, 3000), ("case57.jpg", 1500, 1200), ("case23.jpg", 800, 1500),
         ("case80.jpg", 1000, 2000), ("case42.jpg", 600, 1000)]
models = {n: build_pix(ck, device="cuda", dtype=torch.float16) for n, ck in specs}
order = [n for n, _ in specs[1:]] + [specs[0][0]]
for i, (f, y0, x0) in enumerate(CASES):
    p = os.path.join(TEST, f)
    if not os.path.exists(p): print("skip", f); continue
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    H, W, _ = a.shape; y0 = min(y0, H - 512); x0 = min(x0, W - 512)
    x = torch.from_numpy(a).permute(2, 0, 1)[None].cuda().half() * 2 - 1
    tiles = [a[y0:y0 + 512, x0:x0 + 512]]
    for n in order:
        with torch.no_grad(): y = (models[n].core(x).clamp(-1, 1).float() * 0.5 + 0.5)[0].permute(1, 2, 0).cpu().numpy()
        tiles.append(y[y0:y0 + 512, x0:x0 + 512])
    strip = Image.fromarray((np.concatenate(tiles, 1) * 255).round().astype(np.uint8))
    d = ImageDraw.Draw(strip)
    for k, lab in enumerate(["LQ"] + order): d.rectangle([k * 512, 0, k * 512 + 8 * len(lab) + 8, 16], fill=(0, 0, 0)); d.text((k * 512 + 4, 2), lab, fill=(255, 255, 0))
    strip.save(os.path.join(OUT, f"{i}_{f[:-4]}_{y0}_{x0}.png"))
    print("saved", i, f, y0, x0, flush=True)
print("VIZ_DONE", flush=True)
