"""下载 LSDIR_540 并切成 512 训练 patch。

LSDIR 是 OSEDiff / AdcSR / SamsungAICamera 都在用的标准训练集（8.4 万张）。
danjacobellis/LSDIR_540 是 540p 短边版本，parquet 格式，总约 42GB。

策略：边下边处理，每个 parquet 分片解出图后立即切 512 patch 存 png，
      然后删掉 parquet 省空间。
"""
import os, io, glob, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = "/cache/zihaohe/csig_isr"
OUT = f"{ROOT}/train_data/lsdir512"
TMP = "/cache/zihaohe/tmp/lsdir_parquet"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

N_SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 20   # 下多少个分片
PATCH = 512
PER_IMG = 2          # 每张图切几个 patch
MIN_STD = 0.05       # 跳过纯色/低纹理区域

import pyarrow.parquet as pq
import urllib.request

BASE = "https://hf-mirror.com/datasets/danjacobellis/LSDIR_540/resolve/main/data"
saved = 0
for i in range(N_SHARD):
    name = f"train-{i:05d}-of-00089.parquet"
    lp = os.path.join(TMP, name)
    if not os.path.exists(lp):
        url = f"{BASE}/{name}"
        try:
            urllib.request.urlretrieve(url, lp)
        except Exception as e:
            print(f"  [{i}] 下载失败 {str(e)[:50]}", flush=True)
            continue
    try:
        tb = pq.read_table(lp)
    except Exception as e:
        print(f"  [{i}] 读取失败 {str(e)[:50]}", flush=True)
        os.remove(lp); continue

    col = None
    for c in tb.column_names:
        if c.lower() in ("image", "img", "jpg", "png"):
            col = c; break
    if col is None:
        print(f"  [{i}] 未找到图像列，列名: {tb.column_names}", flush=True)
        os.remove(lp); continue

    rng = np.random.RandomState(i)
    n_this = 0
    for row in tb.column(col):
        v = row.as_py()
        b = v.get("bytes") if isinstance(v, dict) else v
        if not b: continue
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            continue
        a = np.asarray(im, dtype=np.uint8)
        h, w = a.shape[:2]
        if h < PATCH or w < PATCH: continue
        got = 0
        for _ in range(PER_IMG * 4):
            if got >= PER_IMG: break
            y, x = rng.randint(0, h-PATCH+1), rng.randint(0, w-PATCH+1)
            p = a[y:y+PATCH, x:x+PATCH]
            if p.astype(np.float32).std()/255.0 < MIN_STD: continue
            Image.fromarray(p).save(f"{OUT}/ls_{saved:06d}.png")
            saved += 1; got += 1; n_this += 1
    os.remove(lp)
    print(f"  [{i+1}/{N_SHARD}] {name} -> +{n_this} patch, 累计 {saved}", flush=True)

print(f"\n完成，共 {saved} 张 512 patch -> {OUT}")
print(f"占用 {sum(os.path.getsize(f) for f in glob.glob(OUT+'/*.png'))/1e9:.2f} GB")
