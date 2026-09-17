"""幻觉检测器标定。

依据：真实复原只加高频、保留低频；凭空长出物体会改变**局部**低频。
训练里的 w_fid 已经在算低频差，但取的是全图均值，局部幻觉被平均掉了。
所以判据取局部极值。

三个候选量（都不依赖颜色，避免只抓"绿色"这一种表现）：
  lf_p999   低频偏差的 99.9 分位（幻觉是稀疏强信号，均值看不见）
  lf_top    低频偏差最大的 0.1% 像素的均值
  sat_p999  饱和度偏差的 99.9 分位（幻觉物体通常颜色饱和）

用已知的正反例标定：teacher2 严重 / vosr_cap 轻微 / distill_v1、run12 干净。
一个可用的判据必须把这四个按严重程度排对，且干净档与轻微档之间有可分间隔。
"""
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["tinysr", "vosr"], required=True)
ap.add_argument("--out", default="/cache/zihaohe/csig_isr/eval_out")
ap.add_argument("--ncrop", type=int, default=24, help="每张验证图取多少个 512 块")
args = ap.parse_args()

sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
if args.family == "vosr":
    sys.path.insert(0, "/cache/zihaohe/VOSR")
sys.path.insert(0, "/cache/zihaohe")

import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from settings import VAL

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
LQS = ["case1_lq.jpg", "case2_lq.jpg", "case3_lq.jpg"]
TINYSR = [("run12", f"{R}/train_out/run12_iqa30/best.pth"),
          ("mosaic_cap", f"{R}/train_out/mosaic_cap/best.pth"),
          ("distill_v1", f"{R}/train_out/distill_v1/best.pth")]
VOSR = [("teacher_v2", f"{R}/train_out/teacher_koniq_v2/best.pth"),
        ("teacher_v1", f"{R}/train_out/vosr_teacher_koniq/best.pth"),
        ("vosr_cap", f"{R}/train_out/vosr_cap/best.pth"),
        ("vosr_cap2", f"{R}/train_out/vosr_cap2/best.pth"),
        ("teacher2", f"{R}/train_out/vosr_teacher2/best.pth")]


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def sat(x):
    """HSV 的 S 通道，不引依赖直接算"""
    mx = x.max(1, keepdim=True).values
    mn = x.min(1, keepdim=True).values
    return (mx - mn) / mx.clamp_min(1e-6)


def metrics(out, lq, k=16):
    """out/lq: [1,3,H,W] in [0,1]"""
    lo = F.avg_pool2d(out, k); ll = F.avg_pool2d(lq, k)
    dev = (lo - ll).abs().mean(1).flatten()                 # 逐位置低频偏差
    n = dev.numel()
    kk = max(1, int(n * 0.001))
    top = torch.topk(dev, kk).values
    sd = (sat(F.avg_pool2d(out, k)) - sat(F.avg_pool2d(lq, k))).abs().flatten()
    return {"lf_mean": float(dev.mean()),
            "lf_p999": float(torch.quantile(dev.float(), 0.999)),
            "lf_top": float(top.mean()),
            "sat_p999": float(torch.quantile(sd.float(), 0.999))}


def grid(full, n, s=512):
    H, W = full.shape[-2:]
    ny = int(np.ceil(np.sqrt(n * H / max(W, 1))))
    nx = int(np.ceil(n / max(ny, 1)))
    ys = np.linspace(0, H - s, max(ny, 1)).astype(int)
    xs = np.linspace(0, W - s, max(nx, 1)).astype(int)
    return [(int(y), int(x)) for y in ys for x in xs][:n]


cands = TINYSR if args.family == "tinysr" else VOSR
if args.family == "tinysr":
    from tinysr_runtime import build_model

    def get(ck):
        m, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=1.0)
        return m.eval()
else:
    from vosr_load import load_vosr

    def get(ck):
        return load_vosr(ck, DEV, DT, verbose=False)

res = {}
imgs = [load(os.path.join(VAL, n)) for n in LQS]
for tag, ck in cands:
    if not os.path.exists(ck):
        print(f"  缺 {tag}", flush=True); continue
    m = get(ck)
    acc = {k: [] for k in ("lf_mean", "lf_p999", "lf_top", "sat_p999")}
    for full in imgs:
        for (y, x) in grid(full, args.ncrop):
            blk = full[:, :, y:y + 512, x:x + 512].to(DEV, dtype=DT)
            with torch.no_grad(), torch.autocast("cuda", dtype=DT):
                o = m(blk).float().clamp(0, 1)
            mm = metrics(o, blk.float())
            for k in acc: acc[k].append(mm[k])
    # 幻觉是稀疏事件：块级取 95 分位，而不是均值
    res[tag] = {k: float(np.percentile(v, 95)) for k, v in acc.items()}
    res[tag]["lf_top_max"] = float(np.max(acc["lf_top"]))
    print(f"  {tag:<12} lf_mean={res[tag]['lf_mean']:.4f} lf_p999={res[tag]['lf_p999']:.4f} "
          f"lf_top={res[tag]['lf_top']:.4f} lf_top_max={res[tag]['lf_top_max']:.4f} "
          f"sat_p999={res[tag]['sat_p999']:.4f}", flush=True)
    del m; torch.cuda.empty_cache()

p = os.path.join(args.out, f"halluc_{args.family}.json")
json.dump(res, open(p, "w"), indent=2, ensure_ascii=False)
other = os.path.join(args.out, f"halluc_{'vosr' if args.family=='tinysr' else 'tinysr'}.json")
if os.path.exists(other):
    res.update(json.load(open(other)))
    print("\n=== 汇总（目视分档：teacher2 严重 / vosr_cap 轻微 / 其余干净）===")
    print(f"{'模型':<14}{'lf_mean':>10}{'lf_p999':>10}{'lf_top':>10}{'lf_top_max':>12}{'sat_p999':>10}")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["lf_top"]):
        print(f"{k:<14}{v['lf_mean']:>10.4f}{v['lf_p999']:>10.4f}{v['lf_top']:>10.4f}"
              f"{v['lf_top_max']:>12.4f}{v['sat_p999']:>10.4f}")
