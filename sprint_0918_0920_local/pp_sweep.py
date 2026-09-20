"""后处理扫描:同一模型输出上比较若干轻量后处理(可进 runner 的图内算子)对 100 张测试集 NR 与 3 张验证集 FR 的影响。
用法: pp_sweep.py ckpt [--test N]"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code"); sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from settings import VAL, TEST
from metrics_utils import create_metrics
from pix_model import build_pix
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
ck = sys.argv[1]; NTEST = int(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 100
def gauss_blur(x, sigma=2.0, r=4):
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype); g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    return F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), k, groups=3)
def unsharp(x, s, sigma=2.0, r=4): return (x + s * (x - gauss_blur(x, sigma, r))).clamp(0, 1)
def sat(x, a): g = x.mean(1, keepdim=True); return (g + a * (x - g)).clamp(0, 1)
def cas(x, s):  # 3x3 局部对比自适应锐化的简化版:x + s*(x - box3(x))
    return (x + s * (x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1))).clamp(0, 1)
PP = {"none": lambda x: x, "us_r4_s05": lambda x: unsharp(x, 0.05), "us_r4_s10": lambda x: unsharp(x, 0.10), "us_r4_s15": lambda x: unsharp(x, 0.15),
      "us_r2_s10": lambda x: unsharp(x, 0.10, 1.0, 2), "sat108": lambda x: sat(x, 1.08), "cas12": lambda x: cas(x, 0.12), "us10_sat105": lambda x: sat(unsharp(x, 0.10), 1.05)}
def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.; return torch.from_numpy(a).permute(2, 0, 1)[None]
@torch.no_grad()
def infer(m, x01):
    with torch.autocast("cuda", dtype=DT): return m(x01.to(DEV, dtype=DT)).float().clamp(0, 1)
M = create_metrics(DEV); model = build_pix(ck, device=DEV, dtype=DT)
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"), ("case3_lq.jpg", "case3_gt.jpg")]
res = {k: {"fr": [], "nr": {q: [] for q in ("niqe", "musiq", "maniqa", "clipiqa")}} for k in PP}
with torch.no_grad():
    for lqf, gtf in PAIRS:
        sr0 = infer(model, load(os.path.join(VAL, lqf))); gt = load(os.path.join(VAL, gtf)).to(DEV)
        for k, f in PP.items():
            sr = f(sr0)
            with torch.autocast("cuda", dtype=torch.float16):
                lp = float(M["lpips"](sr, gt)); ds = float(M["dists"](sr, gt))
            res[k]["fr"].append((1 - lp) + (1 - ds)); torch.cuda.empty_cache()
    files = sorted(glob.glob(os.path.join(TEST, "*.jpg")), key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))[:NTEST]
    for i, fpath in enumerate(files, 1):
        sr0 = infer(model, load(fpath))
        for k, f in PP.items():
            sr = f(sr0)
            for q in res[k]["nr"]: res[k]["nr"][q].append(float(M[q](sr))); torch.cuda.empty_cache()
        if i % 20 == 0: print(f"  {i}/{len(files)}", flush=True)
print(f"{'pp':12s} {'FR':>7s} {'dFR':>7s} {'testNR':>7s} {'dNR':>7s} {'dS':>7s}  niqe musiq maniqa clipiqa")
base_fr = float(np.mean(res["none"]["fr"])); t = {q: float(np.mean(v)) for q, v in res["none"]["nr"].items()}
base_nr = (10 - t["niqe"]) / 10 + t["musiq"] / 100 + t["maniqa"] + t["clipiqa"]
out = {}
for k in PP:
    fr = float(np.mean(res[k]["fr"])); t = {q: float(np.mean(v)) for q, v in res[k]["nr"].items()}
    nr = (10 - t["niqe"]) / 10 + t["musiq"] / 100 + t["maniqa"] + t["clipiqa"]
    out[k] = {"fr": fr, "nr": nr, **t}
    print(f"{k:12s} {fr:7.4f} {fr-base_fr:+7.4f} {nr:7.4f} {nr-base_nr:+7.4f} {fr+nr-base_fr-base_nr:+7.4f}  {t['niqe']:.3f} {t['musiq']:.2f} {t['maniqa']:.4f} {t['clipiqa']:.4f}", flush=True)
json.dump(out, open("/root/pix_out/logs/pp_sweep_" + os.path.basename(os.path.dirname(ck)) + ".json", "w"), indent=2)
print("PP_SWEEP_DONE", flush=True)
