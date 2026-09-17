"""α 敏感性诊断：v6 在全尺度三张验证图上扫 fixed_alpha。

动机：所有 val 的 tier 分解一直显示 light 档比 heavy 档低 ~0.3
（例：medium 4.19 / light 3.86 / heavy 4.21）。轻退化图占约 1/3，
若 light 档能靠处理强度调制拉回 0.3，整体 +0.1 Score —— 这是
指标直推平台化(4.17)之后剩下的最大结构性杠杆。

本诊断回答：α（残差强度）对逐 case 的 Score 影响。case2 是轻退化(30.1dB)，
若它的最优 α 明显 ≠1.0，就值得训 alpha_gate（逐图自适应，在 runner 内、约+0.1ms）。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
CK = f"{R}/train_out/distill_v6/best.pth"
PAIRS = [("case1_lq.jpg", "case1_gt.png", "中(22.4dB)"),
         ("case2_lq.jpg", "case2_gt.png", "轻(30.1dB)"),
         ("case3_lq.jpg", "case3_gt.jpg", "重(24.8dB)")]


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def tiled(model, x, tile=512, ov=320):
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
            with torch.autocast("cuda", dtype=DT):
                p = model(x[:, :, y:y + tile, xx:xx + tile].to(DEV, dtype=DT)).float()
            out[:, :, y:y + tile, xx:xx + tile] += p * wm
            acc[:, :, y:y + tile, xx:xx + tile] += wm
    return (out / acc.clamp_min(1e-8)).clamp(0, 1)


M = create_metrics(DEV)
print(f"{'alpha':>6} | " + " | ".join(f"{t:>14}" for _, _, t in PAIRS) + " |  平均Score", flush=True)
for alpha in (0.7, 0.85, 1.0, 1.15, 1.3):
    model, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=float(alpha))
    model.eval(); seed_all()
    per = []
    for l, g, _ in PAIRS:
        gt = load(os.path.join(VAL, g)).to(DEV)
        sr = tiled(model, load(os.path.join(VAL, l)))
        v = {}
        for k in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa"):
            v[k] = float(M[k](sr, gt) if k in ("lpips", "dists") else M[k](sr))
        s = ((1 - v["lpips"]) + (1 - v["dists"]) + v["clipiqa"] + v["maniqa"]
             + v["musiq"] / 100.0 + max(0.0, (10 - v["niqe"]) / 10.0))
        per.append(s)
        del sr, gt; torch.cuda.empty_cache()
    print(f"{alpha:>6.2f} | " + " | ".join(f"{s:>14.4f}" for s in per)
          + f" | {np.mean(per):>9.4f}", flush=True)
    del model; torch.cuda.empty_cache()
