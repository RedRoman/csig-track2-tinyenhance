"""Zero-retrain fixed-alpha sweep for an existing TinySR checkpoint.

Use this before training an adaptive gate. It tests the central hypothesis that
light degradations prefer weaker generative residuals while heavy ones benefit
from alpha near/above 1.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch

from settings import ROOT
from metrics_utils import create_metrics, proxy_score
from tinysr_runtime import build_model
from score_eval import collect_crops

DEV = torch.device("cuda")
DT = torch.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alphas", default="0.70,0.78,0.84,0.90,0.96,1.00,1.04,1.08,1.12")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval_out", "alpha_sweep.json"))
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]

    model, report = build_model(args.ckpt, device=DEV, dtype=DT, merge_lora=False,
                                alpha_mode="fixed", fixed_alpha=1.0)
    model.eval()
    print("load report:", report)
    items = collect_crops(args.n)
    M = create_metrics(DEV)
    rows = []

    for a in alphas:
        model.fixed_alpha.fill_(a)
        by_tier = {"heavy": [], "medium": [], "light": []}
        by_case = {"case1": [], "case2": [], "case3": []}
        all_m = {k: [] for k in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")}
        with torch.no_grad():
            for it in items:
                lq, gt = it["lq"].to(DEV, dtype=DT), it["gt"].to(DEV)
                with torch.autocast(device_type="cuda", dtype=DT):
                    out = model(lq).float()
                local = {}
                for k, fn in M.items():
                    if k == "psnr":
                        continue
                    v = fn(out, gt) if k in ("lpips", "dists") else fn(out)
                    local[k] = float(v.item())
                    all_m[k].append(local[k])
                sc = proxy_score(local)
                by_tier[it["tier"]].append(sc)
                by_case[it["case"]].append(sc)
                torch.cuda.empty_cache()
        mean_m = {k: float(np.mean(v)) for k, v in all_m.items()}
        row = {"alpha": a, "score": proxy_score(mean_m), "metrics": mean_m,
               "tier": {k: float(np.mean(v)) for k, v in by_tier.items()},
               "case": {k: float(np.mean(v)) for k, v in by_case.items()}}
        rows.append(row)
        print(f"alpha={a:.3f} score={row['score']:.4f} tier={row['tier']}", flush=True)

    best = max(rows, key=lambda x: x["score"])
    payload = {"checkpoint": args.ckpt, "rows": rows, "best": best}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("best:", best)
    print("saved", args.out)


if __name__ == "__main__":
    main()
