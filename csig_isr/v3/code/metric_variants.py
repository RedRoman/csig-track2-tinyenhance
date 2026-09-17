"""Metric-variant robustness audit for one or more TinySR candidates.

Example:
  python metric_variants.py \
    --candidate run8 /.../tinysr8_fid/best.pth 1.00 \
    --candidate niqe05 /.../niqe_w0.5/best.pth 1.00 \
    --candidate nlong /.../niqe_long/best.pth 1.00

Unlike the previous script, v3 makes the evaluated checkpoint explicit in the
output and can compare all key candidates in one run, preventing stale-candidate
variant reports.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import pyiqa

from settings import ROOT
from eval_utils import collect_crops
from metrics_utils import create_metrics, aggregate_metric_rows, load_metric_names, proxy_score
from tinysr_runtime import build_model

DEV, DT = torch.device("cuda"), torch.float16

VARIANTS = {
    "maniqa":  ["maniqa", "maniqa-koniq", "maniqa-pipal", "maniqa-kadid"],
    "musiq":   ["musiq", "musiq-koniq", "musiq-spaq", "musiq-paq2piq", "musiq-ava"],
    "clipiqa": ["clipiqa", "clipiqa+", "clipiqa+_vitL14_512"],
    "niqe":    ["niqe", "niqe_matlab", "brisque"],
}


def render_candidate(name, ckpt, alpha, items):
    model, rep = build_model(ckpt, device=DEV, dtype=DT, merge_lora=False,
                             alpha_mode="fixed", fixed_alpha=float(alpha))
    model.eval()
    outs = []
    with torch.no_grad():
        for it in items:
            x = it["lq"].to(DEV, dtype=DT)
            with torch.autocast(device_type="cuda", dtype=DT):
                outs.append(model(x).float().clamp(0, 1).cpu())
    del model
    torch.cuda.empty_cache()
    return {"name": name, "ckpt": ckpt, "alpha": float(alpha), "load_report": rep, "outs": outs}


def metric_means(metric, items, candidates, full_reference=False):
    vals = {"lq": [], "gt": []}
    for c in candidates:
        vals[c["name"]] = []
    with torch.no_grad():
        for idx, it in enumerate(items):
            lq = it["lq"].to(DEV)
            gt = it["gt"].to(DEV)
            if full_reference:
                vals["lq"].append(float(metric(lq, gt).mean().item()))
                vals["gt"].append(float(metric(gt, gt).mean().item()))
                for c in candidates:
                    vals[c["name"]].append(float(metric(c["outs"][idx].to(DEV), gt).mean().item()))
            else:
                vals["lq"].append(float(metric(lq).mean().item()))
                vals["gt"].append(float(metric(gt).mean().item()))
                for c in candidates:
                    vals[c["name"]].append(float(metric(c["outs"][idx].to(DEV)).mean().item()))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def canonical_candidate_metrics(items, candidates):
    metrics = create_metrics(DEV, include_psnr=True)
    summaries = {}
    for c in candidates:
        rows = []
        with torch.no_grad():
            for idx, it in enumerate(items):
                out = c["outs"][idx].to(DEV)
                gt = it["gt"].to(DEV)
                m = {}
                for key, fn in metrics.items():
                    v = fn(out, gt) if key in ("lpips", "dists", "psnr") else fn(out)
                    m[key] = float(v.mean().item())
                rows.append({"metrics": m, "case": it["case"], "tier": it["tier"], "alpha": c["alpha"]})
        s = aggregate_metric_rows(rows)
        s.pop("rows", None)
        summaries[c["name"]] = s
    del metrics
    torch.cuda.empty_cache()
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", nargs=3, metavar=("NAME", "CKPT", "ALPHA"),
                    help="repeatable candidate specification")
    # Backward-compatible one-candidate mode.
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out_json", default=f"{ROOT}/eval_out/metric_variants_v3.json")
    args = ap.parse_args()

    specs = list(args.candidate or [])
    if args.ckpt:
        specs.append(["candidate", args.ckpt, str(args.alpha)])
    if not specs:
        raise ValueError("provide at least one --candidate NAME CKPT ALPHA or --ckpt")
    names = [x[0] for x in specs]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")

    items = collect_crops(args.n)
    print(f"crops={len(items)} candidates={names}", flush=True)
    candidates = []
    for name, ckpt, alpha in specs:
        print(f"render {name}: alpha={alpha} ckpt={ckpt}", flush=True)
        candidates.append(render_candidate(name, ckpt, float(alpha), items))

    canonical = canonical_candidate_metrics(items, candidates)
    canonical_names = load_metric_names()
    print("canonical metric names:", canonical_names, flush=True)

    variant_results = {}
    for family, variant_names in VARIANTS.items():
        for metric_name in variant_names:
            try:
                metric = pyiqa.create_metric(metric_name, device=DEV)
            except Exception as e:
                print(f"  skip {metric_name}: {type(e).__name__}: {e}", flush=True)
                continue
            means = metric_means(metric, items, candidates, full_reference=False)
            row = {"family": family, "values": means, "proxy_if_replaced": {}}
            # BRISQUE is a different scale/direction and cannot be inserted into
            # the NIQE slot of the proxy formula. Other same-family variants are
            # reported as sensitivity probes, not organizer-formula claims.
            compatible = metric_name != "brisque"
            row["formula_compatible"] = compatible
            if compatible:
                for c in candidates:
                    base = dict(canonical[c["name"]]["metrics"])
                    base[family] = means[c["name"]]
                    row["proxy_if_replaced"][c["name"]] = proxy_score(base)
            variant_results[metric_name] = row
            msg = "  " + f"{metric_name:26s} lq={means['lq']:.4f} gt={means['gt']:.4f}"
            for c in candidates:
                msg += f" {c['name']}={means[c['name']]:.4f}"
            print(msg, flush=True)
            del metric
            torch.cuda.empty_cache()

    payload = {
        "candidates": [{k: c[k] for k in ("name", "ckpt", "alpha", "load_report")} for c in candidates],
        "metric_names": canonical_names,
        "canonical": canonical,
        "variants": variant_results,
        "note": "Variant replacement scores are sensitivity probes only; the organizer's exact metric variants/formula are unpublished.",
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("saved", args.out_json)


if __name__ == "__main__":
    main()
