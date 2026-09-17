"""在我们自己的口径下评测一个 jit 模型（用于跨队伍可比）。

为什么必须做：我们两个锚点的 Score 几乎相同（3.8723 / 3.8617），
没有杠杆臂，无法检验本地 Score 是否有预测力；而且方向还是反的
（Score 低的榜分反而高）。学弟这个点 Score 低一大截、榜分低 2.8 倍，
是唯一能验证评测器有效性的样本。

协议与 fullres_rank.py 完全一致：3 张全分辨率验证图、tile 512/overlap 256、
高斯融合、MANIQA 采样 100 + 固定种子。
"""
import os, sys, json, time, argparse, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics, proxy_score
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
PAIRS = [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]


def seed_all(s=1234):
    import random
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]


@torch.no_grad()
def tiled_jit(m, x, tile=512, ov=256):
    """jit 模型输入输出都是 [-1,1]，内部转换。"""
    _,_,H,W = x.shape; st = tile-ov
    ys = list(range(0, max(H-tile,0)+1, st)) or [0]
    xs = list(range(0, max(W-tile,0)+1, st)) or [0]
    if ys[-1]+tile < H: ys.append(max(H-tile,0))
    if xs[-1]+tile < W: xs.append(max(W-tile,0))
    t = torch.linspace(-1,1,tile,device=DEV); g = torch.exp(-(t**2)/(2*0.45**2))
    wm = (g[:,None]*g[None,:])[None,None].float()
    out = torch.zeros(1,3,H,W,device=DEV); acc = torch.zeros(1,1,H,W,device=DEV)
    for y in ys:
        for xx in xs:
            p = x[:,:,y:y+tile,xx:xx+tile].to(DEV, dtype=DT)*2-1
            r = m(p.contiguous(memory_format=torch.channels_last))
            r = ((r.float()+1)/2).clamp(0,1)
            out[:,:,y:y+tile,xx:xx+tile] += r*wm
            acc[:,:,y:y+tile,xx:xx+tile] += wm
    return (out/acc.clamp_min(1e-8)).clamp(0,1)


def bench(m, n=50, w=20):
    x = torch.zeros(1,3,512,512, device=DEV, dtype=DT).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        for _ in range(w): m(x)
        torch.cuda.synchronize(); best = 1e9
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(n): m(x)
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter()-t0)/n*1000)
    return best


ap = argparse.ArgumentParser()
ap.add_argument("--jit", required=True)
ap.add_argument("--tag", default="jit")
ap.add_argument("--out_json", default="/cache/zihaohe/csig_isr/eval_out/eval_jit.json")
a = ap.parse_args()

m = torch.jit.load(a.jit, map_location=DEV).eval()
print(f"{a.tag} 载入完成，同机时延 {bench(m):.2f} ms", flush=True)

M = create_metrics(DEV, include_psnr=True)
seed_all(); rows = []
for lqn, gtn in PAIRS:
    lq, gt = load(os.path.join(VAL,lqn)), load(os.path.join(VAL,gtn))
    if lq.shape != gt.shape:
        print(f"  跳过 {lqn}（尺寸不符）"); continue
    sr = tiled_jit(m, lq); gtd = gt.to(DEV); r = {}
    for k in ("lpips","dists"): r[k] = float(M[k](sr, gtd))
    for k in ("niqe","musiq","maniqa","clipiqa"): r[k] = float(M[k](sr))
    r["psnr"] = float(M["psnr"](sr, gtd)); rows.append(r)
    print(f"  {lqn} 完成", flush=True)
    del sr; torch.cuda.empty_cache()

mm = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
mm["score"] = proxy_score(mm); mm["latency_ms"] = bench(m)
K = ["lpips","dists","niqe","musiq","maniqa","clipiqa","psnr","score"]
print()
print(a.tag + "  " + "  ".join(f"{k}={mm[k]:.4f}" for k in K))
json.dump(mm, open(a.out_json,"w"), indent=2)
