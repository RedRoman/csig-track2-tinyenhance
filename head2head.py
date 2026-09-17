"""同场对决：PixelDiT-CSIGv2@240k vs distill_v2b vs teacher_v3，完全相同的 8 张 LQ。

LQ/GT 来自同事的固定评测（CSIGDegradeV2, seed 43+idx, 512px, DIV2K 0801-0808）。
六项官方口径（koniq MANIQA），外加 PSNR 与 lf_top 幻觉指标。
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch, torch.nn.functional as F, random
from PIL import Image
from metrics_utils import create_metrics
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
D = "/tmp/pdit_v2"
M = create_metrics(DEV)


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


bases = sorted(set(f.rsplit("_", 1)[0] for f in os.listdir(D)))
print(f"{len(bases)} 组样本", flush=True)


def six(sr, gt, lq):
    v = {}
    for k in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa"):
        v[k] = float(M[k](sr, gt) if k in ("lpips", "dists") else M[k](sr))
    v["psnr"] = 10 * np.log10(1.0 / max(float(((sr - gt) ** 2).mean()), 1e-12))
    lo = F.avg_pool2d(sr, 16); ll = F.avg_pool2d(lq, 16)
    dev_ = (lo - ll).abs().mean(1).flatten()
    kk = max(1, int(dev_.numel() * 0.001))
    v["lf_top"] = float(torch.topk(dev_, kk).values.mean())
    return v


def agg(rows, tag):
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    m["score"] = ((1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"] + m["maniqa"]
                  + m["musiq"] / 100.0 + max(0.0, (10 - m["niqe"]) / 10.0))
    print(f"{tag:<22} lpips={m['lpips']:.4f} dists={m['dists']:.4f} niqe={m['niqe']:.3f} "
          f"musiq={m['musiq']:.2f} maniqa={m['maniqa']:.4f} clipiqa={m['clipiqa']:.4f} "
          f"psnr={m['psnr']:.2f} lf_top={m['lf_top']:.4f} Score={m['score']:.4f}", flush=True)
    return m


# 1) LQ 基线 与 GT 自身
seed_all()
rows_lq, rows_gt = [], []
for b in bases:
    lq = load(f"{D}/{b}_lq.png").to(DEV); gt = load(f"{D}/{b}_gt.png").to(DEV)
    rows_lq.append(six(lq, gt, lq)); rows_gt.append(six(gt, gt, lq))
agg(rows_lq, "LQ 基线")
agg(rows_gt, "GT 自身")

# 2) PixelDiT SR（同事产出）
seed_all()
rows = []
for b in bases:
    sr = load(f"{D}/{b}_sr.png").to(DEV)
    gt = load(f"{D}/{b}_gt.png").to(DEV); lq = load(f"{D}/{b}_lq.png").to(DEV)
    rows.append(six(sr, gt, lq))
agg(rows, "PixelDiT-CSIGv2@240k")

# 3) 我们的模型在同样 LQ 上（512 原生，直接单次前向）
CANDS = [("distill_v2b(已提交)", f"{R}/train_out/distill_v2b/best.pth", "tinysr"),
         ("distill_v3c(现最佳)", f"{R}/train_out/distill_v3c_iqa15/best.pth", "tinysr")]
for tag, ck, fam in CANDS:
    model, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=1.0)
    model.eval(); seed_all()
    rows = []
    for b in bases:
        lq = load(f"{D}/{b}_lq.png"); gt = load(f"{D}/{b}_gt.png").to(DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=DT):
            sr = model(lq.to(DEV, dtype=DT)).float().clamp(0, 1)
        rows.append(six(sr, gt, lq.to(DEV)))
    agg(rows, tag)
    del model; torch.cuda.empty_cache()
