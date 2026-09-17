"""给一个装着 case1/2/3 输出图的目录打全尺度六项分。与 VOSR 那次同一口径。"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")
GTS = ["case1_gt.png","case2_gt.png","case3_gt.jpg"]
M = create_metrics(DEV)

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None].to(DEV)

def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)

for d in sys.argv[1:]:
    seed_all()
    acc = {k: [] for k in ("lpips","dists","niqe","musiq","maniqa","clipiqa")}
    ok = True
    for i, g in enumerate(GTS, 1):
        cand = [os.path.join(d, f"case{i}{e}") for e in (".png",".jpg",".jpeg")]
        p = next((c for c in cand if os.path.exists(c)), None)
        if p is None: print(f"  {d}: 缺 case{i}"); ok = False; break
        sr = load(p); gt = load(os.path.join(VAL, g))
        if sr.shape != gt.shape:
            print(f"  {d}: case{i} 尺寸 {tuple(sr.shape[-2:])} != GT {tuple(gt.shape[-2:])}"); ok=False; break
        for k in acc:
            acc[k].append(float(M[k](sr, gt) if k in ("lpips","dists") else M[k](sr)))
        del sr, gt; torch.cuda.empty_cache()
    if not ok: continue
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    score = ((1-m["lpips"]) + (1-m["dists"]) + m["clipiqa"] + m["maniqa"]
             + m["musiq"]/100.0 + max(0.0, (10-m["niqe"])/10.0))
    m["score"] = score
    print(f"{os.path.basename(d.rstrip('/')):<16} lpips={m['lpips']:.4f} dists={m['dists']:.4f} "
          f"niqe={m['niqe']:.4f} musiq={m['musiq']:.4f} maniqa={m['maniqa']:.4f} "
          f"clipiqa={m['clipiqa']:.4f} score={score:.4f}", flush=True)
    json.dump(m, open(os.path.join("/cache/zihaohe/csig_isr/eval_out",
              f"scoredir_{os.path.basename(d.rstrip('/'))}.json"), "w"), indent=2)
