"""全分辨率候选排序器 —— 唯一可信的选型依据。

用法: python fullres_rank.py DIR_OR_CKPT [...] [--alpha 1.0]
目录会自动展开 best.pth / final.pth / snap_*.pth。

为什么必须用它：512 crop 口径已被证明会选错（crop 第一名在全尺度是倒数第一，
MUSIQ 在两个尺度上符号相反）。训练里的 best.pth 是 crop 选出来的，不可信。

MANIQA 的 forward 是 random_crop(x,224,crop_num=test_sample)，在 4K 图上只抽
少量小块；真实图空间差异大，Score 抖动可达 0.011，且共用全局 RNG 会让结果
依赖评测顺序。故评测端提高采样数 + 每候选前固定种子。
"""
import os, sys, glob, json, argparse, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
import numpy as np, torch
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics, proxy_score
from tinysr_runtime import build_model
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
PAIRS = [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]


def seed_all(seed: int = 1234):
    import random
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def tiled(model, x, tile=512, ov=256):
    _, _, H, W = x.shape; st = tile - ov
    ys = list(range(0, max(H-tile, 0)+1, st)) or [0]
    xs = list(range(0, max(W-tile, 0)+1, st)) or [0]
    if ys[-1]+tile < H: ys.append(max(H-tile, 0))
    if xs[-1]+tile < W: xs.append(max(W-tile, 0))
    t = torch.linspace(-1, 1, tile, device=DEV); g = torch.exp(-(t**2)/(2*0.45**2))
    wm = (g[:, None]*g[None, :])[None, None].float()
    out = torch.zeros(1, 3, H, W, device=DEV); acc = torch.zeros(1, 1, H, W, device=DEV)
    for y in ys:
        for xx in xs:
            with torch.autocast(device_type="cuda", dtype=DT):
                p = model(x[:, :, y:y+tile, xx:xx+tile].to(DEV, dtype=DT)).float()
            out[:, :, y:y+tile, xx:xx+tile] += p*wm
            acc[:, :, y:y+tile, xx:xx+tile] += wm
    return (out/acc.clamp_min(1e-8)).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="+")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out_json", default="/cache/zihaohe/csig_isr/eval_out/fullres_rank.json")
    a = ap.parse_args()

    cks = []
    for t in a.targets:
        if os.path.isdir(t):
            cand = [os.path.join(t, "best.pth"), os.path.join(t, "final.pth")]
            cand += sorted(glob.glob(os.path.join(t, "snap_*.pth")))
            for f in cand:
                if os.path.exists(f):
                    cks.append((f"{os.path.basename(t)}/{os.path.basename(f)}", f))
        elif os.path.exists(t):
            cks.append((os.path.basename(os.path.dirname(t))+"/"+os.path.basename(t), t))
    print(f"待评 {len(cks)} 个 checkpoint\n", flush=True)

    M = create_metrics(DEV, include_psnr=True)
    imgs = [(load(os.path.join(VAL, l)), load(os.path.join(VAL, g))) for l, g in PAIRS]
    res = {}
    for tag, ck in cks:
        try:
            model, rep = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                                     alpha_mode="fixed", fixed_alpha=a.alpha)
        except Exception as e:
            print(f"  {tag}: 载入失败 {type(e).__name__}: {str(e)[:80]}"); continue
        model.eval(); rows = []
        seed_all()
        for lq, gt in imgs:
            if lq.shape != gt.shape: continue
            sr = tiled(model, lq); gtd = gt.to(DEV); r = {}
            for k in ("lpips", "dists"): r[k] = float(M[k](sr, gtd))
            for k in ("niqe", "musiq", "maniqa", "clipiqa"): r[k] = float(M[k](sr))
            r["psnr"] = float(M["psnr"](sr, gtd)); rows.append(r)
            del sr; torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()
        if not rows: continue
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        m["score"] = proxy_score(m); m["step"] = rep.get("step"); res[tag] = m
        print(f"  {tag:<34} Score {m['score']:.4f}  PSNR {m['psnr']:.2f}", flush=True)

    K = ["lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa", "psnr", "score"]
    print()
    print("候选".ljust(32) + "".join(f"{k:>9}" for k in K))
    print("-"*(32+9*len(K)))
    for tag, m in sorted(res.items(), key=lambda kv: -kv[1]["score"]):
        print(tag.ljust(32) + "".join(f"{m[k]:>9.4f}" for k in K))
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(res, open(a.out_json, "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved {a.out_json}")


if __name__ == "__main__":
    main()
