"""把 viz_tiles.py 出的块拼成对比图：每行一个场景，列按 LQ → 学生 → 教师 → GT 排。"""
import os, glob, sys
from PIL import Image, ImageDraw

D = "/cache/zihaohe/viz_tiles"
OUT = "/cache/zihaohe/viz_compare.png"
files = sorted(glob.glob(os.path.join(D, "*.png")))
scenes, cols = [], []
for f in files:
    b = os.path.basename(f)[:-4]
    sc, col = b.split("__")
    if sc not in scenes: scenes.append(sc)
    if col not in cols: cols.append(col)
cols.sort()
scenes.sort()
print("场景:", scenes)
print("列:", cols)

S = Image.open(files[0]).size[0]
pad, hdr, lab = 5, 24, 96
Wt = lab + len(cols) * S + (len(cols) + 1) * pad
Ht = hdr + len(scenes) * (S + pad) + pad
canvas = Image.new("RGB", (Wt, Ht), (22, 22, 22))
d = ImageDraw.Draw(canvas)
for c, col in enumerate(cols):
    x0 = lab + pad + c * (S + pad)
    d.text((x0 + 4, 6), col.split("_", 1)[1] if "_" in col else col, fill=(245, 245, 245))
for r, sc in enumerate(scenes):
    y0 = hdr + r * (S + pad)
    d.text((6, y0 + S // 2), sc, fill=(200, 200, 200))
    for c, col in enumerate(cols):
        p = os.path.join(D, f"{sc}__{col}.png")
        if not os.path.exists(p):
            continue
        canvas.paste(Image.open(p), (lab + pad + c * (S + pad), y0))
canvas.save(OUT, quality=95)
print("已存", OUT, canvas.size)
