"""对已渲染的验证集出图打全六项分（与 fullres_rank.py 完全同口径）。"""
import os, sys, json, glob, argparse, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics, proxy_score
Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")
GT = {"case1": "case1_gt.png", "case2": "case2_gt.png", "case3": "case3_gt.jpg"}

def seed_all(s=1234):
    import random
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--tag", default="cand")
a = ap.parse_args()

M = create_metrics(DEV, include_psnr=True)
seed_all(); rows = []
for case, gtn in GT.items():
    hit = glob.glob(os.path.join(a.dir, f"{case}*"))
    if not hit: print(f"  缺 {case}"); continue
    sr = load(hit[0]).to(DEV); gt = load(os.path.join(VAL, gtn)).to(DEV)
    if sr.shape != gt.shape:
        print(f"  {case} 尺寸不符 {tuple(sr.shape[-2:])} vs {tuple(gt.shape[-2:])}"); continue
    r = {}
    for k in ("lpips","dists"): r[k] = float(M[k](sr, gt))
    for k in ("niqe","musiq","maniqa","clipiqa"): r[k] = float(M[k](sr))
    r["psnr"] = float(M["psnr"](sr, gt)); rows.append(r)
    print(f"  {case} 完成", flush=True)
    del sr, gt; torch.cuda.empty_cache()

m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
m["score"] = proxy_score(m)
K = ["lpips","dists","niqe","musiq","maniqa","clipiqa","psnr","score"]
print()
print(a.tag + "  " + "  ".join(f"{k}={m[k]:.4f}" for k in K))
json.dump(m, open(f"/cache/zihaohe/csig_isr/eval_out/{a.tag}.json","w"), indent=2)
