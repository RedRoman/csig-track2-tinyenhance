"""同一 crop 过多个配置，横向对比伪纹理从哪一档开始出现。"""
import os, sys, numpy as np, cv2, torch
from PIL import Image, ImageDraw
from settings import ROOT
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
LQ_DIR = os.path.join(ROOT, "data", "赛题二", "测试集")
DST = os.path.join(ROOT, "review_final"); os.makedirs(DST, exist_ok=True)
C = 512

# (标签, checkpoint, alpha)
CFGS = [
    ("run8 submitted 71.54", f"{ROOT}/train_out/tinysr8_fid/best.pth",   1.00),
    ("soup1.80 a=1.20",      f"{ROOT}/train_out/v2_soup_hi/soup_1.80.pth", 1.20),
    ("niqe_w0.5 a=1.00",     f"{ROOT}/train_out/niqe_w0.5/best.pth",     1.00),
    ("niqe_long a=1.00",     f"{ROOT}/train_out/niqe_long/best.pth",     1.00),
]
# (文件名, x, y)  —— 取自 mk_review.py 挑出的最高 lapvar 位置
SPOTS = [("case24.jpg", 512, 1792), ("case25.jpg", 768, 512), ("case16.jpg", 1024, 2048)]

crops = {}
for nm, x, y in SPOTS:
    a = np.asarray(Image.open(os.path.join(LQ_DIR, nm)).convert("RGB"), np.float32) / 255.0
    crops[nm] = torch.from_numpy(a[y:y+C, x:x+C]).permute(2, 0, 1)[None]

cols = {nm: [(f"LQ 输入", (crops[nm][0].permute(1,2,0).numpy()*255).astype(np.uint8))] for nm, _, _ in SPOTS}
for tag, ck, al in CFGS:
    if not os.path.exists(ck):
        print("跳过（无 ckpt）:", tag); continue
    m, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                       alpha_mode="fixed", fixed_alpha=al)
    m.eval()
    with torch.no_grad():
        for nm, _, _ in SPOTS:
            with torch.autocast(device_type="cuda", dtype=DT):
                o = m(crops[nm].to(DEV, dtype=DT)).float().clamp(0, 1)
            cols[nm].append((tag, (o[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))
    del m; torch.cuda.empty_cache()
    print("done", tag, flush=True)

for nm, _, _ in SPOTS:
    items = cols[nm]; n = len(items)
    cv = np.full((C + 30, n*C + (n-1)*8, 3), 255, np.uint8)
    for i, (tag, img) in enumerate(items):
        cv[30:, i*(C+8):i*(C+8)+C] = img
    im = Image.fromarray(cv); d = ImageDraw.Draw(im)
    for i, (tag, _) in enumerate(items):
        d.text((i*(C+8)+6, 9), f"{nm}  {tag}", fill=(0,0,0) if i == 0 else (180,0,0))
    f = os.path.join(DST, f"cmp_{nm.replace('.jpg','')}.png"); im.save(f); print(f)
