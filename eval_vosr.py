"""给微调后的 VOSR checkpoint 测与 TinySR 完全同口径的两个判据：
  A. 3 张验证图、原生 4K 全尺度六项 Score
  B. 100 张真实测试图、原生 4K 四项 NR
用法: eval_vosr.py 标签=best.pth路径 [...] [--test N]
"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch, random
from PIL import Image
from settings import VAL, TEST
from metrics_utils import create_metrics

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]
args_in = [a for a in sys.argv[1:] if not a.startswith("--")]
NTEST = 100
if "--test" in sys.argv:
    NTEST = int(sys.argv[sys.argv.index("--test") + 1])

sys.path.insert(0, "/cache/zihaohe")
from vosr_load import load_vosr


def load_wrapper(spec):
    """spec 可以是官方 ckpt 目录，也可以是我们的 best.pth —— 见 vosr_load.py"""
    return load_vosr(spec, DEV, DT)


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
tfiles = sorted(glob.glob(os.path.join(TEST, "*.jpg")),
                key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))[:NTEST]
out_all = {}
for spec_s in args_in:
    tag, ck = spec_s.split("=", 1)
    if not os.path.exists(ck):
        print(f"  缺 {tag} -> {ck}", flush=True); continue
    print(f"=== {tag} ===", flush=True)
    model = load_wrapper(ck)
    seed_all()
    acc = {k: [] for k in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")}
    ps = []
    for l, g in PAIRS:
        gt = load(os.path.join(VAL, g)).to(DEV)
        sr = tiled(model, load(os.path.join(VAL, l)))
        for k in acc:
            acc[k].append(float(M[k](sr, gt) if k in ("lpips", "dists") else M[k](sr)))
        ps.append(10 * np.log10(1.0 / max(float(((sr - gt) ** 2).mean()), 1e-12)))
        del sr, gt; torch.cuda.empty_cache()
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    m["psnr"] = float(np.mean(ps))
    m["score"] = ((1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"] + m["maniqa"]
                  + m["musiq"] / 100.0 + max(0.0, (10 - m["niqe"]) / 10.0))
    print("  [验证集全尺度] lpips={:.4f} dists={:.4f} niqe={:.3f} musiq={:.2f} maniqa={:.4f} "
          "clipiqa={:.4f} psnr={:.2f} Score={:.4f}".format(
              m["lpips"], m["dists"], m["niqe"], m["musiq"], m["maniqa"],
              m["clipiqa"], m["psnr"], m["score"]), flush=True)

    seed_all()
    a = {k: [] for k in ("niqe", "musiq", "maniqa", "clipiqa")}
    for i, f in enumerate(tfiles):
        sr = tiled(model, load(f))
        for k in a: a[k].append(float(M[k](sr)))
        del sr; torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"      测试集 {i+1}/{len(tfiles)}", flush=True)
    t = {k: float(np.mean(v)) for k, v in a.items()}
    t["nr"] = (t["clipiqa"] + t["maniqa"] + t["musiq"] / 100.0
               + max(0.0, (10 - t["niqe"]) / 10.0))
    print("  [测试集100张] niqe={:.3f} musiq={:.2f} maniqa={:.4f} clipiqa={:.4f} NR={:.4f}".format(
        t["niqe"], t["musiq"], t["maniqa"], t["clipiqa"], t["nr"]), flush=True)
    out_all[tag] = {"val_fullres": m, "test_nr": t}
    del model; torch.cuda.empty_cache()
    json.dump(out_all, open(f"{R}/eval_out/eval_vosr.json", "w"), indent=2, ensure_ascii=False)

print()
print("对照 TinySR：combo NR 2.6144→榜72.60 | ref1 2.5986 | run12 2.5250/Score4.0025 | run8 2.4535→榜71.54")
