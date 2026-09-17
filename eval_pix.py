"""像素 UNet 全尺度评测:验证集 3 张六项(整图直推,无需分块)+ 测试集 100 张 NR 四项。用法: eval_pix.py 标签=ckpt [--test N]"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code"); sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch, random
from PIL import Image
from settings import VAL, TEST
from metrics_utils import create_metrics
from pix_model import build_pix
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"), ("case3_lq.jpg", "case3_gt.jpg")]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NTEST = int(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 100
def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]
@torch.no_grad()
def infer(model, x01):
    with torch.autocast("cuda", dtype=DT):
        return model(x01.to(DEV, dtype=DT)).float().clamp(0, 1)
M = create_metrics(DEV)
tfiles = sorted(glob.glob(os.path.join(TEST, "*.jpg")), key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))[:NTEST]
out = {}
for spec in args:
    tag, ck = spec.split("=", 1)
    model = build_pix(ck, device=DEV, dtype=DT)
    rows = []
    for lqf, gtf in PAIRS:
        sr = infer(model, load(os.path.join(VAL, lqf))); gt = load(os.path.join(VAL, gtf)).to(DEV)
        v = {}
        with torch.autocast("cuda", dtype=torch.float16):
            for k in ("lpips", "dists"): v[k] = float(M[k](sr, gt)); torch.cuda.empty_cache()
        for k in ("niqe", "musiq", "maniqa", "clipiqa"): v[k] = float(M[k](sr)); torch.cuda.empty_cache()
        v["psnr"] = 10 * np.log10(1.0 / max(float(((sr - gt) ** 2).mean()), 1e-12)); rows.append(v)
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    m["score"] = (1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"] + m["maniqa"] + m["musiq"] / 100 + max(0, (10 - m["niqe"]) / 10)
    print(f"  {tag:<12} [验证集全尺度] lpips={m['lpips']:.4f} dists={m['dists']:.4f} niqe={m['niqe']:.3f} musiq={m['musiq']:.2f} "
          f"maniqa={m['maniqa']:.4f} clipiqa={m['clipiqa']:.4f} psnr={m['psnr']:.2f} Score={m['score']:.4f}", flush=True)
    nr = {k: [] for k in ("niqe", "musiq", "maniqa", "clipiqa")}
    for i, f in enumerate(tfiles, 1):
        sr = infer(model, load(f))
        for k in nr: nr[k].append(float(M[k](sr))); torch.cuda.empty_cache()
        if i % 25 == 0: print(f"      {tag} 测试集 {i}/{len(tfiles)}", flush=True)
    t = {k: float(np.mean(v)) for k, v in nr.items()}
    t["nr"] = (10 - t["niqe"]) / 10 + t["musiq"] / 100 + t["maniqa"] + t["clipiqa"]
    print(f"  {tag:<12} [测试集NR] nr={t['nr']:.4f} niqe={t['niqe']:.3f} musiq={t['musiq']:.2f} maniqa={t['maniqa']:.4f} clipiqa={t['clipiqa']:.4f}", flush=True)
    out[tag] = {"val_fullres": m, "test_nr": t}
    del model; torch.cuda.empty_cache()
os.makedirs(f"{R}/eval_out", exist_ok=True)
json.dump(out, open(f"{R}/eval_out/eval_pix_{args[0].split('=')[0]}.json", "w"), indent=2)
print("EVAL_PIX_DONE", flush=True)
