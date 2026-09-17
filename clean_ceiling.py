"""实验 B：真实高清原图的 NR 水位 —— 无幻觉的自然上界。

我们一直在问「NR 能推到多高」，从没问过「不作弊能到多高」。
已知验证集 GT（手机原图）NR 只有 1.975，而 teacher2（整片仙人掌）到 3.293。
中间哪一段是真细节、哪一段是编造的，决定了整个优化目标是否成立。

测四组，全部原生分辨率、与所有历史数字同口径：
  LSDIR 原图          真·高清自然图像
  验证集 GT           我们这批数据的真值
  GT + 轻度锐化       只用合法操作能加多少
  GT + 强锐化         合法操作的极限
"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F, random
from PIL import Image
from settings import VAL, ROOT
from metrics_utils import create_metrics

Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")
KEYS = ("niqe", "musiq", "maniqa", "clipiqa")
M = create_metrics(DEV)


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def nr(t):
    v = {k: float(M[k](t)) for k in KEYS}
    v["nr"] = (v["clipiqa"] + v["maniqa"] + v["musiq"] / 100.0
               + max(0.0, (10 - v["niqe"]) / 10.0))
    return v


def unsharp(x, amount, radius=2.0):
    """标准非锐化掩模：只放大已有的高频，不引入任何新内容"""
    k = int(radius * 4) | 1
    xs = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-(xs ** 2) / (2 * radius ** 2)); g = (g / g.sum()).view(1, 1, 1, k)
    b = F.conv2d(F.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"), g.expand(3, 1, 1, k), groups=3)
    b = F.conv2d(F.pad(b, (0, 0) + (k // 2,) * 2, mode="reflect"),
                 g.transpose(2, 3).expand(3, 1, k, 1), groups=3)
    return (x + amount * (x - b)).clamp(0, 1)


def summarize(tag, rows):
    m = {k: float(np.mean([r[k] for r in rows])) for k in list(rows[0])}
    print("{:<26} niqe={:.3f} musiq={:.2f} maniqa={:.4f} clipiqa={:.4f}  NR={:.4f}".format(
        tag, m["niqe"], m["musiq"], m["maniqa"], m["clipiqa"], m["nr"]), flush=True)
    return m


out = {}
GTS = ["case1_gt.png", "case2_gt.png", "case3_gt.jpg"]
gts = [load(os.path.join(VAL, g)).to(DEV) for g in GTS]

seed_all(); out["验证集GT"] = summarize("验证集 GT（真值）", [nr(t) for t in gts])
for amt in (0.5, 1.0, 2.0):
    seed_all()
    out[f"GT+锐化{amt}"] = summarize(f"GT + 非锐化掩模 {amt}", [nr(unsharp(t, amt)) for t in gts])

# LSDIR 原图：用全分辨率原始文件，不是我们裁好的 512
cands = sorted(glob.glob(os.path.join(ROOT, "AdcSR/testset/DIV2K-Val/HR/*.png")))[:12]
if cands:
    seed_all()
    rows = []
    for p in cands:
        t = load(p).to(DEV)
        rows.append(nr(t)); del t; torch.cuda.empty_cache()
    out["DIV2K原图"] = summarize(f"DIV2K-Val 原图（{len(cands)}张）", rows)

ls = sorted(glob.glob(os.path.join(ROOT, "train_data", "lsdir512", "*.png")))[:24]
if ls:
    seed_all()
    rows = []
    for p in ls:
        t = load(p).to(DEV)
        rows.append(nr(t)); del t; torch.cuda.empty_cache()
    out["LSDIR512原图"] = summarize(f"LSDIR 512 原图（{len(ls)}张）", rows)

print()
print("对照：distill_v1 测试集 NR 2.6960 | vosr_cap 2.9524 | teacher2(整片仙人掌) 3.293 | 原始LQ 1.1992")
json.dump(out, open(f"{ROOT}/eval_out/clean_ceiling.json", "w"), indent=2, ensure_ascii=False)
