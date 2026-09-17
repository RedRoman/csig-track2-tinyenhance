"""GT-aware high-frequency hallucination diagnostics for TinySR candidates.

This script does not affect training. It quantifies whether a candidate injects
more high-frequency energy than GT specifically in GT-smooth regions, and ranks
validation crops by overshoot severity. Optional heatmaps make the failure mode
visually inspectable before enabling --w_hf_over.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch

from settings import ROOT
from eval_utils import collect_crops
from hf_regularization import hf_diagnostic_maps, highpass_luma
from tinysr_runtime import build_model

DEV, DT = torch.device("cuda"), torch.float16


def spectral_peakiness(x: torch.Tensor) -> float:
    """Simple periodic-texture diagnostic: high-percentile FFT power / mean power.

    This is a heuristic only. It is useful comparatively (SR vs GT on the same
    crop), not as an absolute perceptual-quality metric.
    """
    hp = highpass_luma(x.float(), blur_kernel=5)[0, 0]
    f = torch.fft.fftshift(torch.fft.fft2(hp))
    p = f.abs().square()
    h, w = p.shape
    yy = torch.arange(h, device=p.device)[:, None] - h // 2
    xx = torch.arange(w, device=p.device)[None, :] - w // 2
    rr = torch.sqrt(yy.float().square() + xx.float().square())
    keep = rr > 0.08 * min(h, w)
    pv = p[keep]
    if pv.numel() == 0:
        return 0.0
    return float((torch.quantile(pv, 0.995) / (pv.mean() + 1e-12)).item())


def save_heatmap(path: str, excess: torch.Tensor):
    a = excess.detach().float().cpu().numpy()[0, 0]
    hi = float(np.percentile(a, 99.5))
    if hi <= 1e-12:
        norm = np.zeros_like(a, dtype=np.uint8)
    else:
        norm = np.clip(a / hi * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, heat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", nargs=3, metavar=("NAME", "CKPT", "ALPHA"), required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--blur_kernel", type=int, default=5)
    ap.add_argument("--local_kernel", type=int, default=9)
    ap.add_argument("--margin", type=float, default=0.35)
    ap.add_argument("--slack", type=float, default=2e-4)
    ap.add_argument("--smooth_quantile", type=float, default=0.35)
    ap.add_argument("--mask_softness", type=float, default=0.20)
    ap.add_argument("--smooth_abs_cap", type=float, default=0.0)
    ap.add_argument("--topk_maps", type=int, default=0,
                    help="save excess heatmaps for top-K crops per candidate")
    ap.add_argument("--map_dir", default=os.path.join(ROOT, "eval_out", "hf_maps"))
    ap.add_argument("--out_json", default=os.path.join(ROOT, "eval_out", "hf_diagnostics.json"))
    a = ap.parse_args()

    items = collect_crops(a.n)
    output = {"settings": vars(a), "candidates": {}}

    for name, ckpt, alpha_s in a.candidate:
        alpha = float(alpha_s)
        model, rep = build_model(ckpt, device=DEV, dtype=DT, merge_lora=False,
                                 alpha_mode="fixed", fixed_alpha=alpha)
        model.eval()
        rows = []
        with torch.no_grad():
            for idx, it in enumerate(items):
                lq = it["lq"].to(DEV, dtype=DT)
                gt = it["gt"].to(DEV).float()
                with torch.autocast(device_type="cuda", dtype=DT):
                    sr = model(lq).float().clamp(0, 1)
                maps = hf_diagnostic_maps(
                    sr, gt,
                    blur_kernel=a.blur_kernel,
                    local_kernel=a.local_kernel,
                    margin=a.margin,
                    slack=a.slack,
                    smooth_quantile=a.smooth_quantile,
                    mask_softness=a.mask_softness,
                    smooth_abs_cap=a.smooth_abs_cap,
                )
                mask = maps["smooth_mask"]
                denom = mask.sum().clamp_min(1.0)
                sr_s = (maps["sr_energy"] * mask).sum() / denom
                gt_s = (maps["gt_energy"] * mask).sum() / denom
                excess = (maps["excess"] * mask).sum() / denom
                over_frac = (((maps["excess"] > 0).float() * mask).sum() / denom)
                row = {
                    "index": idx,
                    "case": it["case"],
                    "tier": it["tier"],
                    "xy": it.get("xy"),
                    "psnr_lq": float(it["psnr_lq"]),
                    "hf_global_ratio": float((maps["sr_energy"].mean() / (maps["gt_energy"].mean() + 1e-8)).item()),
                    "hf_smooth_ratio": float((sr_s / (gt_s + 1e-8)).item()),
                    "hf_overshoot": float(excess.item()),
                    "overshoot_fraction": float(over_frac.item()),
                    "spectral_peakiness_sr": spectral_peakiness(sr),
                    "spectral_peakiness_gt": spectral_peakiness(gt),
                }
                row["spectral_peakiness_ratio"] = row["spectral_peakiness_sr"] / max(row["spectral_peakiness_gt"], 1e-8)
                rows.append(row)
                # Temporarily keep map tensor for optional top-k saving.
                row["_excess"] = maps["excess"].cpu()
                torch.cuda.empty_cache()

        # Rank by smooth-region overshoot, the failure mode we care about.
        ranked = sorted(rows, key=lambda r: r["hf_overshoot"], reverse=True)
        if a.topk_maps > 0:
            for rank, r in enumerate(ranked[:a.topk_maps], 1):
                fn = f"{name}_rank{rank:02d}_{r['case']}_{r['tier']}_idx{r['index']:02d}.png"
                save_heatmap(os.path.join(a.map_dir, fn), r["_excess"])
        for r in rows:
            r.pop("_excess", None)

        summary_keys = ["hf_global_ratio", "hf_smooth_ratio", "hf_overshoot",
                        "overshoot_fraction", "spectral_peakiness_ratio"]
        summary = {k: float(np.mean([r[k] for r in rows])) for k in summary_keys}
        by_tier = {}
        for tier in ("heavy", "medium", "light"):
            rr = [r for r in rows if r["tier"] == tier]
            if rr:
                by_tier[tier] = {k: float(np.mean([r[k] for r in rr])) for k in summary_keys}
        output["candidates"][name] = {
            "checkpoint": ckpt,
            "alpha": alpha,
            "load_report": rep,
            "summary": summary,
            "tier": by_tier,
            "top_overshoot": ranked[:min(10, len(ranked))],
            "rows": rows,
        }
        print(f"{name}: smooth HF ratio={summary['hf_smooth_ratio']:.3f} "
              f"overshoot={summary['hf_overshoot']:.6f} "
              f"periodic-ratio={summary['spectral_peakiness_ratio']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("saved", a.out_json)


if __name__ == "__main__":
    main()
