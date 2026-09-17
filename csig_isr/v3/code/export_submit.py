"""Export one explicitly selected TinySR v3 candidate to TorchScript.

The exporter and 4K renderer share tinysr_runtime.build_model, so a checkpoint can
no longer silently point at a different run. A SHA256 manifest is emitted beside
the model.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image

from settings import ROOT, TEST
from tinysr_runtime import build_model, SubmitModel
from candidate_manifest import sha256_file, write_manifest, git_commit
from metrics_utils import load_metric_names

DEV = torch.device("cuda")
DT = torch.float16


def bench(fn, x, n=60, warm=20):
    with torch.no_grad():
        for _ in range(warm):
            fn(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn(x)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def sample_input():
    path = os.path.join(TEST, "case1.jpg")
    if os.path.exists(path):
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1)[None]
        _, _, H, W = t.shape
        y, x = max(0, H//2-256), max(0, W//2-256)
        t = t[:, :, y:y+512, x:x+512]
        if t.shape[-2:] == (512, 512):
            return (t * 2 - 1).to(DEV, dtype=DT)
    return torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "submit", "model_dir"))
    ap.add_argument("--alpha_mode", choices=["auto", "fixed", "adaptive"], default="auto")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--osediff_ms", type=float, default=0.0,
                    help="optional same-platform OSEDiff latency; 0 means do not report speedup")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    core, report = build_model(args.ckpt, device=DEV, dtype=DT, merge_lora=True,
                               alpha_mode=args.alpha_mode, fixed_alpha=args.alpha)
    # Competition runner requires fp16 weights. This also converts the tiny gate.
    core = core.half().eval()
    model = SubmitModel(core).to(DEV).eval()
    x = sample_input()

    with torch.no_grad():
        ref = model(x).float().clone()
    t_eager = bench(model, x)
    print("load report:", report)
    print(f"eager {t_eager:.3f} ms range=[{ref.min().item():.3f},{ref.max().item():.3f}]")

    with torch.no_grad():
        traced = torch.jit.trace(model, x, check_trace=False).eval()
        try:
            frozen = torch.jit.freeze(traced, optimize_numerics=False)
            freeze_tag = "freeze(optimize_numerics=False)"
        except Exception:
            frozen = traced
            freeze_tag = "trace-only"

    model_path = os.path.join(args.out_dir, "tinysr_x1.pt")
    frozen.save(model_path)
    print(f"saved {model_path} {os.path.getsize(model_path)/1e6:.1f} MB via {freeze_tag}")

    del frozen, traced, model, core
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    loaded = torch.jit.load(model_path, map_location=DEV).eval()
    with torch.no_grad():
        out = loaded(x).float()
    diff = (out - ref).abs()
    t_jit = bench(loaded, x)
    rel = diff.mean().item() / max(ref.abs().mean().item(), 1e-8)
    print(f"jit {t_jit:.3f} ms diff max={diff.max().item():.7f} mean={diff.mean().item():.9f} rel={rel*100:.5f}%")

    with torch.no_grad():
        o1 = loaded(x).float().clone()
        _ = loaded(torch.rand_like(x) * 2 - 1)
        o2 = loaded(x).float().clone()
    state_diff = float((o1-o2).abs().max().item())
    print(f"state pollution max diff={state_diff:.9f}")

    # Copy the single source config beside the JIT artifact.
    cfg_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
    if os.path.exists(cfg_src):
        shutil.copy2(cfg_src, os.path.join(args.out_dir, "config.yaml"))

    speedup = (args.osediff_ms / t_jit) if args.osediff_ms > 0 else None
    result = {"t_eager_ms": t_eager, "t_jit_ms": t_jit, "speedup": speedup,
              "rel_diff": rel, "max_diff": float(diff.max().item()),
              "state_pollution": state_diff, "checkpoint": args.ckpt,
              "checkpoint_sha256": sha256_file(args.ckpt),
              "jit_sha256": sha256_file(model_path), "loader_report": report,
              "alpha_mode": args.alpha_mode, "effective_alpha_mode": report.get("alpha_mode"), "alpha": args.alpha,
              "metric_names": load_metric_names()}
    with open(os.path.join(args.out_dir, "export_verify.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_manifest(os.path.join(args.out_dir, "candidate_manifest.json"),
                   candidate_type="jit_export", git_commit=git_commit(os.path.dirname(__file__)), **result)
    print("checkpoint SHA256", result["checkpoint_sha256"])
    print("JIT SHA256", result["jit_sha256"])


if __name__ == "__main__":
    main()
