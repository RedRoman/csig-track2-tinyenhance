"""Canonical TinySR proxy evaluator for CSIG track-2 v2.

Unlike the legacy script, this file actually builds TinySR checkpoints and uses
exactly the same metric names as training/tile/inference audit.
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from settings import ROOT, VAL
from metrics_utils import create_metrics, proxy_score, load_metric_names
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")
DT = torch.float16
PAIRS = [("case1_lq.jpg", "case1_gt.png", 101, "case1"),
         ("case2_lq.jpg", "case2_gt.png", 202, "case2"),
         ("case3_lq.jpg", "case3_gt.jpg", 303, "case3")]


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def collect_crops(n_per=12):
    out = []
    for lqn, gtn, seed, case in PAIRS:
        lq_f, gt_f = load_img(os.path.join(VAL, lqn)), load_img(os.path.join(VAL, gtn))
        if lq_f.shape != gt_f.shape:
            continue
        _, _, H, W = gt_f.shape
        rng = np.random.RandomState(seed)
        got, tries = 0, 0
        while got < n_per and tries < n_per * 20:
            tries += 1
            y, x = rng.randint(0, max(H - 512, 1)), rng.randint(0, max(W - 512, 1))
            gt = gt_f[:, :, y:y + 512, x:x + 512]
            if gt.shape[-2:] != (512, 512) or gt.std().item() < 0.035:
                continue
            lq = lq_f[:, :, y:y + 512, x:x + 512]
            psnr_lq = -10.0 * np.log10(F.mse_loss(lq, gt).item() + 1e-12)
            out.append({"lq": lq.clone(), "gt": gt.clone(), "case": case,
                        "psnr_lq": psnr_lq})
            got += 1
    ps = sorted(x["psnr_lq"] for x in out)
    lo, hi = ps[len(ps)//3], ps[2*len(ps)//3]
    for x in out:
        x["tier"] = "heavy" if x["psnr_lq"] < lo else ("light" if x["psnr_lq"] > hi else "medium")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--lq_only", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="fixed alpha when --alpha_mode fixed")
    ap.add_argument("--alpha_mode", choices=["auto", "fixed", "adaptive"], default="auto")
    ap.add_argument("--robust_beta", type=float, default=0.15)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    print("metrics:", load_metric_names())
    items = collect_crops(args.n)
    print(f"validation crops={len(items)}")
    M = create_metrics(DEV, include_psnr=True)

    model = None
    if not args.lq_only:
        model, report = build_model(args.ckpt, device=DEV, dtype=DT, merge_lora=False,
                                    alpha_mode=args.alpha_mode, fixed_alpha=args.alpha)
        model.eval()
        print("load report:", report)

    vals = defaultdict(list)
    case_scores, tier_scores, alphas = defaultdict(list), defaultdict(list), []
    with torch.no_grad():
        for it in items:
            lq, gt = it["lq"].to(DEV), it["gt"].to(DEV)
            if args.lq_only:
                out, alpha = lq, torch.ones(1, 1, 1, 1, device=DEV)
            else:
                with torch.autocast(device_type="cuda", dtype=DT):
                    out, alpha = model.forward_train(lq.to(dtype=DT))
                out = out.float()
            m = {}
            for k, fn in M.items():
                v = fn(out, gt) if k in ("lpips", "dists", "psnr") else fn(out)
                m[k] = float(v.item())
                vals[k].append(m[k])
            sc = proxy_score(m)
            case_scores[it["case"]].append(sc)
            tier_scores[it["tier"]].append(sc)
            alphas.append(float(alpha.float().mean().item()))
            torch.cuda.empty_cache()

    mean = {k: float(np.mean(v)) for k, v in vals.items()}
    score = proxy_score(mean)
    case = {k: float(np.mean(v)) for k, v in case_scores.items()}
    tier = {k: float(np.mean(v)) for k, v in tier_scores.items()}
    robust = score - args.robust_beta * float(np.std(list(case.values())))
    result = {"metrics": mean, "score": score, "robust_score": robust,
              "case": case, "tier": tier,
              "alpha_mean": float(np.mean(alphas)), "alpha_std": float(np.std(alphas)),
              "metric_names": load_metric_names(), "checkpoint": args.ckpt,
              "alpha_mode": args.alpha_mode, "alpha": args.alpha}

    print("\n" + "=" * 72)
    print(f"score={score:.4f} robust={robust:.4f} PSNR={mean['psnr']:.3f}")
    print(f"LPIPS={mean['lpips']:.4f} DISTS={mean['dists']:.4f} "
          f"CLIPIQA={mean['clipiqa']:.4f} MANIQA={mean['maniqa']:.4f} "
          f"MUSIQ={mean['musiq']:.2f} NIQE={mean['niqe']:.4f}")
    print(f"case={case}\ntier={tier}\nalpha={result['alpha_mean']:.3f}±{result['alpha_std']:.3f}")

    out_json = args.out_json or os.path.join(ROOT, "eval_out", "v2_score.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("saved", out_json)


if __name__ == "__main__":
    main()
