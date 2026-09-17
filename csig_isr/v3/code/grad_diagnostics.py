"""Measure per-objective gradient norms on the same TinySR batch.

Use this before increasing NIQE weights. The reported ``canonical_*`` terms are
scaled/sign-corrected as they enter the local proxy objective, so their gradient
norms are directly comparable at first order (direction conflicts are not).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from degradation import degrade
from hf_regularization import gt_aware_hf_overshoot_loss
from metrics_utils import create_metrics
from settings import ROOT
from tinysr_runtime import build_model

DEV, DT = torch.device("cuda"), torch.float16


def prepare_pair(path, seed=0, size=512):
    rng = np.random.RandomState(seed)
    gt = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = gt.shape[:2]
    if h < size or w < size:
        from cv2 import resize, INTER_CUBIC
        gt = resize(gt, (max(w, size), max(h, size)), interpolation=INTER_CUBIC)
        h, w = gt.shape[:2]
    y = int(rng.randint(0, h - size + 1)) if h > size else 0
    x = int(rng.randint(0, w - size + 1)) if w > size else 0
    gt = np.ascontiguousarray(gt[y:y+size, x:x+size])
    lq = degrade(gt, rng)
    to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return to_t(lq).to(DEV), to_t(gt).to(DEV)


def trainable_lora(model):
    ps = []
    for name, p in model.tr.named_parameters():
        p.requires_grad_("lora_" in name)
        if p.requires_grad:
            ps.append(p)
    for p in model.vae.parameters():
        p.requires_grad_(False)
    if not ps:
        raise RuntimeError("no LoRA parameters found; use a non-merged LoRA checkpoint for this diagnostic")
    return ps


def norm_of_grads(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().square().sum().item())
    return total ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default=os.path.join(ROOT, "eval_out", "grad_diagnostics.json"))
    a = ap.parse_args()

    path = a.gt
    if not path and a.gt_dir:
        files = sorted(glob.glob(os.path.join(a.gt_dir, "*.png")) + glob.glob(os.path.join(a.gt_dir, "*.jpg")))
        path = files[0] if files else None
    if not path:
        raise ValueError("provide --gt or --gt_dir")

    lq, gt = prepare_pair(path, a.seed)
    model, rep = build_model(a.ckpt, device=DEV, dtype=DT, merge_lora=False,
                             alpha_mode="fixed", fixed_alpha=1.0)
    params = trainable_lora(model)
    metrics = create_metrics(DEV, as_loss=True)
    model.train()

    names = ["l1", "fid", "canonical_lpips", "canonical_dists", "canonical_clipiqa",
             "canonical_maniqa", "canonical_musiq", "canonical_niqe", "hf_over"]
    result = {"checkpoint": a.ckpt, "load_report": rep, "gt": path, "components": {}}

    for name in names:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=DT):
            out, _ = model.forward_train(lq.to(dtype=DT))
        out = out.float()
        if name == "l1":
            loss = F.l1_loss(out, gt)
        elif name == "fid":
            loss = F.l1_loss(F.avg_pool2d(out, 16), F.avg_pool2d(lq, 16))
        elif name == "canonical_lpips":
            loss = metrics["lpips"](out, gt).mean()
        elif name == "canonical_dists":
            loss = metrics["dists"](out, gt).mean()
        elif name == "canonical_clipiqa":
            loss = -metrics["clipiqa"](out).mean()
        elif name == "canonical_maniqa":
            loss = -metrics["maniqa"](out).mean()
        elif name == "canonical_musiq":
            loss = -metrics["musiq"](out).mean() / 100.0
        elif name == "canonical_niqe":
            loss = metrics["niqe"](out.clamp(0, 1)).mean() / 10.0
        elif name == "hf_over":
            loss, _ = gt_aware_hf_overshoot_loss(out, gt)
        else:
            raise AssertionError(name)
        loss.backward()
        gn = norm_of_grads(params)
        result["components"][name] = {"value": float(loss.detach().item()), "grad_norm": float(gn)}
        print(f"{name:20s} value={float(loss.detach()): .6f} grad_norm={gn:.6e}", flush=True)
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("saved", a.out_json)


if __name__ == "__main__":
    main()
