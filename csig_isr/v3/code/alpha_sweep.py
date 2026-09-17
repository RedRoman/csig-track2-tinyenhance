"""Zero-retrain fixed-alpha sweep using the canonical v3 evaluator."""
from __future__ import annotations

import argparse
import json
import os

import torch

from settings import ROOT
from metrics_utils import create_metrics
from tinysr_runtime import build_model
from eval_utils import collect_crops, evaluate_model

DEV = torch.device("cuda")
DT = torch.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alphas", default="0.70,0.78,0.84,0.90,0.94,0.96,1.00,1.04,1.08,1.12,1.20")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--robust_beta", type=float, default=0.15)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval_out", "alpha_sweep_v3.json"))
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]

    model, report = build_model(args.ckpt, device=DEV, dtype=DT, merge_lora=False,
                                alpha_mode="fixed", fixed_alpha=1.0)
    model.eval()
    print("load report:", report)
    items = collect_crops(args.n)
    metrics = create_metrics(DEV, include_psnr=True)
    rows = []

    for alpha in alphas:
        model.fixed_alpha.fill_(alpha)
        ev = evaluate_model(model, items, metrics, device=DEV, dtype=DT,
                            robust_beta=args.robust_beta)
        ev.pop("rows", None)
        row = {"alpha": alpha, **ev}
        rows.append(row)
        print(f"alpha={alpha:.3f} score={row['score']:.4f} robust={row['robust_score']:.4f} "
              f"PSNR={row['metrics']['psnr']:.3f} tier={row['tier']}", flush=True)

    best = max(rows, key=lambda x: x["score"])
    payload = {"checkpoint": args.ckpt, "rows": rows, "best": best}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("best:", {k: best[k] for k in ("alpha", "score", "robust_score")})
    print("saved", args.out)


if __name__ == "__main__":
    main()
