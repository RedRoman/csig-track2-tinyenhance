"""Shared validation crop construction and canonical TinySR evaluation."""
from __future__ import annotations

import os
from typing import List, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from metrics_utils import aggregate_metric_rows
from settings import VAL

Image.MAX_IMAGE_PIXELS = None

PAIRS = [
    ("case1_lq.jpg", "case1_gt.png", 101, "case1"),
    ("case2_lq.jpg", "case2_gt.png", 202, "case2"),
    ("case3_lq.jpg", "case3_gt.jpg", 303, "case3"),
]


def load_img(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def collect_crops(n_per: int = 12, crop_size: int = 512, val_root: str = VAL,
                  min_std: float = 0.035) -> List[dict]:
    out = []
    for lqn, gtn, seed, case in PAIRS:
        lq_f, gt_f = load_img(os.path.join(val_root, lqn)), load_img(os.path.join(val_root, gtn))
        if lq_f.shape != gt_f.shape:
            continue
        _, _, h, w = gt_f.shape
        rng = np.random.RandomState(seed)
        got, tries = 0, 0
        while got < n_per and tries < n_per * 20:
            tries += 1
            max_y = h - crop_size
            max_x = w - crop_size
            if max_y < 0 or max_x < 0:
                break
            y = int(rng.randint(0, max_y + 1)) if max_y > 0 else 0
            x = int(rng.randint(0, max_x + 1)) if max_x > 0 else 0
            gt = gt_f[:, :, y:y + crop_size, x:x + crop_size]
            if gt.shape[-2:] != (crop_size, crop_size) or gt.std().item() < min_std:
                continue
            lq = lq_f[:, :, y:y + crop_size, x:x + crop_size]
            psnr_lq = -10.0 * np.log10(F.mse_loss(lq, gt).item() + 1e-12)
            out.append({
                "lq": lq.clone(), "gt": gt.clone(), "case": case,
                "psnr_lq": float(psnr_lq), "xy": [x, y],
            })
            got += 1
    if not out:
        raise RuntimeError(f"no validation crops were collected from {val_root}")
    ps = sorted(float(x["psnr_lq"]) for x in out)
    lo, hi = ps[len(ps) // 3], ps[2 * len(ps) // 3]
    for x in out:
        x["tier"] = "heavy" if x["psnr_lq"] < lo else ("light" if x["psnr_lq"] > hi else "medium")
    return out


@torch.no_grad()
def evaluate_model(model, items: List[Mapping], metrics, *, device, dtype=torch.float16,
                   robust_beta: float = 0.15, lq_only: bool = False,
                   empty_cache: bool = True) -> dict:
    """Evaluate with exactly the same aggregation used by score_eval.py.

    Backbone inference may use autocast; all IQA metric forwards run in fp32.
    """
    was_train = None if model is None else model.training
    if model is not None:
        model.eval()
    rows = []
    for it in items:
        lq = it["lq"].to(device)
        gt = it["gt"].to(device)
        if lq_only:
            out = lq.float()
            alpha_value = 1.0
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                out, alpha = model.forward_train(lq.to(dtype=dtype))
            out = out.float().clamp(0, 1)
            alpha_value = float(alpha.float().mean().item())
        m = {}
        for key, fn in metrics.items():
            if key in ("lpips", "dists", "psnr"):
                v = fn(out, gt)
            else:
                v = fn(out)
            m[key] = float(v.mean().item())
        rows.append({
            "metrics": m,
            "case": it.get("case"),
            "tier": it.get("tier"),
            "alpha": alpha_value,
            "psnr_lq": float(it.get("psnr_lq", float("nan"))),
            "xy": it.get("xy"),
        })
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if model is not None and was_train:
        model.train()
    result = aggregate_metric_rows(rows, robust_beta=robust_beta)
    result["rows"] = rows
    return result
