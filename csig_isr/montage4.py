"""四方拼图：LQ | Samsung训练后 | TinySR | GT"""
import os
from PIL import Image, ImageDraw
ROOT="/cache/zihaohe/csig_isr"; D=f"{ROOT}/visualizations/x1_compare"
tags=sorted({f.split("_")[0]+"_"+f.split("_")[1] for f in os.listdir(D) if f.endswith(".png")})
cols=[("1_LQ","LQ 输入"),("4_Samsung","Samsung 训练后"),("3_TinySR","TinySR zero-shot"),("2_GT","GT 参考")]
rows=[]
for t in tags:
    r=[]
    for suf,lab in cols:
        p=f"{D}/{t}_{suf}.png"
        if os.path.exists(p): r.append((p,lab))
    if len(r)==4: rows.append((t,r))
print(f"完整组 {len(rows)}: {[t for t,_ in rows]}")
if not rows: raise SystemExit("无完整组")
W=460; pad,lh=5,22
cv=Image.new("RGB",(4*W+5*pad, len(rows)*(W+lh+pad)+pad),(18,18,18))
d=ImageDraw.Draw(cv)
for ri,(t,r) in enumerate(rows):
    y=pad+ri*(W+lh+pad)
    for ci,(p,lab) in enumerate(r):
        x=pad+ci*(W+pad)
        cv.paste(Image.open(p).convert("RGB").resize((W,W)),(x,y+lh))
        d.text((x+4,y+5),f"{t}  {lab}",fill=(245,245,245))
out=f"{ROOT}/visualizations/four_way_montage.png"
cv.save(out); print(f"已保存 {out} ({cv.width}x{cv.height})")
