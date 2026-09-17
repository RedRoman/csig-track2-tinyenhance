"""拼并排对比图：LQ | TinySR+adain | GT，附放大细节区。"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"
D=f"{ROOT}/visualizations/x1_compare"
OUT=f"{ROOT}/visualizations"

groups={}
for f in sorted(os.listdir(D)):
    if not f.endswith(".png"): continue
    parts=f.rsplit("_",2)
    key=f.split("_")[0]+"_"+f.split("_")[1]
    groups.setdefault(key,{})[f.split("_")[3].replace(".png","")]=os.path.join(D,f)

rows=[]
for key in sorted(groups):
    g=groups[key]
    if not all(k in g for k in ("LQ","GT")): continue
    order=[("LQ","LQ 输入"),("TinySR","TinySR 原样"),("adain","TinySR+adain"),("GT","GT 参考")]
    imgs=[(Image.open(g[k]).convert("RGB"),lab) for k,lab in order if k in g]
    if len(imgs)<3: continue
    rows.append((key,imgs))

if not rows:
    print("无可拼接组"); sys.exit()

W=imgs[0][0].width
pad, labelh = 6, 26
cols=max(len(r[1]) for r in rows)
canvas=Image.new("RGB",(cols*W+(cols+1)*pad, len(rows)*(W+labelh+pad)+pad),(24,24,24))
d=ImageDraw.Draw(canvas)
for ri,(key,imgs) in enumerate(rows):
    y=pad+ri*(W+labelh+pad)
    for ci,(im,lab) in enumerate(imgs):
        x=pad+ci*(W+pad)
        canvas.paste(im.resize((W,W)),(x,y+labelh))
        d.text((x+4,y+6),f"{key}  {lab}",fill=(235,235,235))
p=f"{OUT}/x1_montage.png"
canvas.save(p)
print(f"已保存 {p}  ({canvas.width}x{canvas.height})")
