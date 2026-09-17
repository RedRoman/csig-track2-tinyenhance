"""A1：量化四项无参考指标的「尺度错位」。

训练在 512 块上算 IQA，官方在原生 4K 上评。已知 MUSIQ 在两个尺度上会翻转
（GT 自己：4K 上 40.67，/4 上 72.04），但 MANIQA / CLIPIQA / NIQE 从未测过。

对每个模型的整图输出，在 /1 /2 /4 /8 四个尺度上测四项，另附「直接 512 前向」
（训练时看到的口径）。要回答两个问题：
  Q1 每一项指标的绝对值随图像尺寸怎么变？
  Q2 模型之间的排序在哪一档翻转？翻转的那些项，就是训练目标里必须改口径的。
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F, random
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
LQS = ["case1_lq.jpg", "case2_lq.jpg", "case3_lq.jpg"]
GTS = ["case1_gt.png", "case2_gt.png", "case3_gt.jpg"]
CKS = [("run8", f"{R}/train_out/tinysr8_fid/best.pth"),
       ("run12", f"{R}/train_out/run12_iqa30/best.pth"),
       ("ref1", f"{R}/train_out/ref1/best.pth"),
       ("sepctrl", f"{R}/train_out/sep_control/best.pth")]
KEYS = ("niqe", "musiq", "maniqa", "clipiqa")
SCALES = [1, 2, 4, 8]


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def tiled(model, x, tile=512, ov=256):
    _, _, H, W = x.shape; st = tile - ov
    ys = list(range(0, max(H - tile, 0) + 1, st)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, st)) or [0]
    if ys[-1] + tile < H: ys.append(max(H - tile, 0))
    if xs[-1] + tile < W: xs.append(max(W - tile, 0))
    t = torch.linspace(-1, 1, tile, device=DEV); g = torch.exp(-(t ** 2) / (2 * 0.45 ** 2))
    wm = (g[:, None] * g[None, :])[None, None].float()
    out = torch.zeros(1, 3, H, W, device=DEV); acc = torch.zeros(1, 1, H, W, device=DEV)
    for y in ys:
        for xx in xs:
            with torch.autocast(device_type="cuda", dtype=DT):
                p = model(x[:, :, y:y + tile, xx:xx + tile].to(DEV, dtype=DT)).float()
            out[:, :, y:y + tile, xx:xx + tile] += p * wm
            acc[:, :, y:y + tile, xx:xx + tile] += wm
    return (out / acc.clamp_min(1e-8)).clamp(0, 1)


M = create_metrics(DEV)


def nr4(t):
    return {k: float(M[k](t)) for k in KEYS}


def curve(t):
    """同一张图在 /1 /2 /4 /8 上的四项"""
    o = {}
    for s in SCALES:
        u = t if s == 1 else F.interpolate(t, scale_factor=1.0 / s, mode="bicubic",
                                           align_corners=False, antialias=True).clamp(0, 1)
        o[s] = nr4(u)
    return o


def avg(dicts):
    return {s: {k: float(np.mean([d[s][k] for d in dicts])) for k in KEYS} for s in SCALES}


rows = {}
lq_t = [load(os.path.join(VAL, n)) for n in LQS]
gt_t = [load(os.path.join(VAL, n)) for n in GTS]

for tag, src in (("LQ", lq_t), ("GT", gt_t)):
    seed_all()
    rows[tag] = {"scales": avg([curve(t.to(DEV)) for t in src]), "direct512": None}
    print("  {} 完成".format(tag), flush=True)

for tag, ck in CKS:
    if not os.path.exists(ck):
        print("  缺 {}".format(tag), flush=True); continue
    model, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=1.0)
    model.eval(); seed_all()
    cs, ds = [], []
    for lq in lq_t:
        H, W = lq.shape[-2:]
        c = lq[:, :, H // 2 - 256:H // 2 + 256, W // 2 - 256:W // 2 + 256].to(DEV, dtype=DT)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DT):
            oc = model(c).float().clamp(0, 1)
        ds.append(nr4(oc))
        sr = tiled(model, lq)
        cs.append(curve(sr))
        del sr, oc; torch.cuda.empty_cache()
    rows[tag] = {"scales": avg(cs),
                 "direct512": {k: float(np.mean([d[k] for d in ds])) for k in KEYS}}
    print("  {} 完成".format(tag), flush=True)
    del model; torch.cuda.empty_cache()

json.dump(rows, open(f"{R}/eval_out/scale_mismatch.json", "w"), indent=2, ensure_ascii=False)

for k in KEYS:
    print()
    print("=== {} ({}) ===".format(k, "越低越好" if k == "niqe" else "越高越好"))
    print("{:<10}{:>10}{:>10}{:>10}{:>10}{:>13}".format("", "/1(4K)", "/2", "/4", "/8", "直接512前向"))
    for tag, v in rows.items():
        line = "{:<10}".format(tag)
        for s in SCALES:
            line += "{:>10.3f}".format(v["scales"][s][k])
        line += "{:>13}".format("-" if v["direct512"] is None
                                else "{:.3f}".format(v["direct512"][k]))
        print(line)
    # 排序是否翻转
    mods = [t for t in rows if rows[t]["direct512"] is not None]
    hi = (k != "niqe")
    def rank(vals):
        return [m for m in sorted(mods, key=lambda m: vals(m), reverse=hi)]
    r1 = rank(lambda m: rows[m]["scales"][1][k])
    r8 = rank(lambda m: rows[m]["scales"][8][k])
    rd = rank(lambda m: rows[m]["direct512"][k])
    print("  原生4K排序   {}".format(" > ".join(r1)))
    print("  /8排序       {}".format(" > ".join(r8)))
    print("  直接512排序  {}   {}".format(" > ".join(rd),
          "<<< 与原生4K不一致" if rd != r1 else "(一致)"))
