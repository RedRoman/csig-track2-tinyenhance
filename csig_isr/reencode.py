"""按输入图的编码参数重新保存输出，消除一切可能的格式差异。

对齐项：
  - 色度采样 4:2:0（输入为 h=2,v=2；原输出用 subsampling=0 即 4:4:4，导致膨胀 2.56x）
  - quality 降到 95（视觉无损，文件大小与输入同量级）
  - baseline（非 progressive），与输入一致
  - 不写 EXIF/ICC，避免额外元数据

不重新推理，只是对已生成的像素重编码。
"""
import os, glob, shutil
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = "/cache/zihaohe/csig_isr"
SRC = f"{ROOT}/submit/output_dir"
DST = f"{ROOT}/submit/output_dir_v2"
INP = f"{ROOT}/data/赛题二/测试集"
os.makedirs(DST, exist_ok=True)

files = sorted(glob.glob(f"{SRC}/*.jpg"),
               key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
print(f"重编码 {len(files)} 张 ...")
so = si = sn = 0
bad = []
for i, p in enumerate(files, 1):
    n = os.path.basename(p)
    im = Image.open(p).convert("RGB")
    ip = os.path.join(INP, n)
    ref = Image.open(ip)
    if im.size != ref.size:
        bad.append((n, im.size, ref.size))
    so += os.path.getsize(p)
    si += os.path.getsize(ip)
    out = os.path.join(DST, n)
    # subsampling=2 即 4:2:0，与输入一致；quality 95 视觉无损
    im.save(out, format="JPEG", quality=98, subsampling=2,
            optimize=True, progressive=False)
    sn += os.path.getsize(out)
    if i % 25 == 0:
        print(f"  {i}/{len(files)}")

print()
print(f"原输出 {so/1e6:>8.1f} MB  平均 {so/1e6/len(files):.2f} MB")
print(f"新输出 {sn/1e6:>8.1f} MB  平均 {sn/1e6/len(files):.2f} MB")
print(f"输入   {si/1e6:>8.1f} MB  平均 {si/1e6/len(files):.2f} MB")
print(f"新输出/输入 = {sn/si:.2f}x")
print(f"尺寸异常: {bad or '无'}")

# 校验新图的编码参数与输入一致
import struct
def jinfo(p):
    d = open(p, "rb").read()
    i = 2
    while i < len(d) - 1:
        if d[i] != 0xFF:
            i += 1; continue
        m = d[i + 1]
        if m in (0xC0, 0xC1, 0xC2):
            nc = d[i + 9]
            comps = [(d[i+10+c*3], d[i+11+c*3] >> 4, d[i+11+c*3] & 0xF) for c in range(nc)]
            return ("progressive" if m == 0xC2 else "baseline"), comps
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2; continue
        i += 2 + struct.unpack(">H", d[i+2:i+4])[0]
    return None, None

print()
for tag, p in [("输入", f"{INP}/case1.jpg"), ("新输出", f"{DST}/case1.jpg")]:
    pr, c = jinfo(p)
    print(f"  {tag}: {pr}  分量 {c}")

# 视觉损失核验：新旧输出的 PSNR（应 >45dB 即肉眼无差）
import random
random.seed(0)
tot = 0; k = 0
for p in random.sample(files, 8):
    n = os.path.basename(p)
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(os.path.join(DST, n)).convert("RGB"), dtype=np.float32)
    mse = ((a - b) ** 2).mean()
    tot += 10 * np.log10(255.0 ** 2 / max(mse, 1e-9)); k += 1
print(f"\n重编码前后 PSNR = {tot/k:.2f} dB（>45 视为肉眼无损）")
