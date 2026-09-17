"""Canonical TinySR proxy evaluator for CSIG track-2 v3.

Training validation imports the same ``collect_crops`` and ``evaluate_model``
functions, so checkpoint selection and final local scoring now share one exact
aggregation path.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from settings import ROOT
from metrics_utils import create_metrics, load_metric_names
from eval_utils import collect_crops, evaluate_model
from tinysr_runtime import build_model

DEV = torch.device("cuda")
DT = torch.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--lq_only", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="fixed alpha when --alpha_mode fixed")
    ap.add_argument("--alpha_mode", choices=["auto", "fixed", "adaptive"], default="auto")
    ap.add_argument("--robust_beta", type=float, default=0.15)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat deterministic evaluation to expose unexpected runtime nondeterminism")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >=1")
    if not args.lq_only and not args.ckpt:
        raise ValueError("--ckpt is required unless --lq_only is used")

    print("metrics:", load_metric_names())
    items = collect_crops(args.n)
    print(f"validation crops={len(items)}")
    metrics = create_metrics(DEV, include_psnr=True)

    model = None
    report = None
    if not args.lq_only:
        model, report = build_model(args.ckpt, device=DEV, dtype=DT, merge_lora=False,
                                    alpha_mode=args.alpha_mode, fixed_alpha=args.alpha)
        model.eval()
        print("load report:", report)

    repeat_results = []
    for r in range(args.repeats):
        result = evaluate_model(model, items, metrics, device=DEV, dtype=DT,
                                robust_beta=args.robust_beta, lq_only=args.lq_only)
        # Per-crop rows make JSON very large and are not needed for standard score files.
        result.pop("rows", None)
        repeat_results.append(result)
        m = result["metrics"]
        print("\n" + "=" * 72)
        print(f"repeat={r+1}/{args.repeats} score={result['score']:.4f} "
              f"robust={result['robust_score']:.4f} PSNR={m['psnr']:.3f}")
        print(f"LPIPS={m['lpips']:.4f} DISTS={m['dists']:.4f} "
              f"CLIPIQA={m['clipiqa']:.4f} MANIQA={m['maniqa']:.4f} "
              f"MUSIQ={m['musiq']:.2f} NIQE={m['niqe']:.4f}")
        print(f"case={result['case']}\ntier={result['tier']}\n"
              f"alpha={result['alpha_mean']:.3f}±{result['alpha_std']:.3f}")
        print(f"mean_crop_score(diagnostic only)={result['mean_crop_score_diagnostic']:.4f}")

    scores = [x["score"] for x in repeat_results]
    payload = {
        "checkpoint": args.ckpt,
        "load_report": report,
        "metric_names": load_metric_names(),
        "alpha_mode": args.alpha_mode,
        "alpha": args.alpha,
        "repeats": repeat_results,
        "score_mean": float(sum(scores) / len(scores)),
        "score_range": float(max(scores) - min(scores)),
    }
    # Preserve convenient top-level fields for old downstream scripts. When
    # repeats>1 this is the first deterministic pass; score_mean/range above
    # expose any unexpected runtime variance explicitly.
    payload.update(repeat_results[0])

    out_json = args.out_json or os.path.join(ROOT, "eval_out", "v3_score.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("saved", out_json)


if __name__ == "__main__":
    main()
