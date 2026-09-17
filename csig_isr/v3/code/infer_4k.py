"""Canonical 4K submission rendering for TinySR v3.

No default checkpoint is allowed: every render must explicitly name its source.
This prevents the old run8/run12 drift.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from settings import ROOT, TEST, DEFAULT_TILE, DEFAULT_OVERLAP, TS
from metrics_utils import create_metrics, nr_proxy_contrib, load_metric_names
from tinysr_runtime import build_model
from candidate_manifest import sha256_file, write_manifest, git_commit

# TinySR utility import after runtime has configured sys.path.
from utils.wavelet_color_fix import wavelet_color_fix

Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")
DT = torch.float16


def blend_w(n, sigma=0.45):
    t = torch.linspace(-1, 1, n, device=DEV)
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return (g[:, None] * g[None, :])[None, None].float()


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def tiled(model, img01, tile, overlap):
    _, _, H, W = img01.shape
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile")
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < H:
        ys.append(max(H - tile, 0))
    if not xs or xs[-1] + tile < W:
        xs.append(max(W - tile, 0))
    wm = blend_w(tile)
    out = torch.zeros(1, 3, H, W, device=DEV, dtype=torch.float32)
    acc = torch.zeros(1, 1, H, W, device=DEV, dtype=torch.float32)
    for y in ys:
        for x in xs:
            patch = img01[:, :, y:y+tile, x:x+tile].to(DEV, dtype=DT)
            with torch.autocast(device_type="cuda", dtype=DT):
                pred = model(patch).float()
            out[:, :, y:y+tile, x:x+tile] += pred * wm
            acc[:, :, y:y+tile, x:x+tile] += wm
    return (out / acc.clamp_min(1e-8)).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "submit", "output_dir"))
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--colorfix", choices=["wavelet", "none"], default="wavelet")
    ap.add_argument("--quality", type=int, default=97)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha_mode", choices=["auto", "fixed", "adaptive"], default="auto")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--skip_nr", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, report = build_model(args.ckpt, device=DEV, dtype=DT, merge_lora=True,
                                alpha_mode=args.alpha_mode, fixed_alpha=args.alpha)
    model.eval()
    print("load report:", report, flush=True)
    print("metric config:", load_metric_names(), flush=True)

    files = sorted(glob.glob(os.path.join(TEST, "*.jpg")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
    if args.limit:
        files = files[:args.limit]
    print(f"render {len(files)} images tile={args.tile} overlap={args.overlap} "
          f"alpha_mode={args.alpha_mode} alpha={args.alpha}", flush=True)

    NR = None if args.skip_nr else create_metrics(DEV)
    stats = {"lq": {k: [] for k in ("niqe", "musiq", "maniqa", "clipiqa")},
             "sr": {k: [] for k in ("niqe", "musiq", "maniqa", "clipiqa")}}
    t0 = time.time()
    output_hashes = {}
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        lq = load_img(f)
        sr = tiled(model, lq, args.tile, args.overlap)
        if args.colorfix == "wavelet":
            fixed = wavelet_color_fix(ToPILImage()(sr[0].cpu()), ToPILImage()(lq[0].cpu()))
            sr = ToTensor()(fixed)[None].to(DEV)
        arr = (sr[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        out_path = os.path.join(args.out, name)
        Image.fromarray(arr).save(out_path, quality=args.quality, subsampling=0)
        output_hashes[name] = sha256_file(out_path)

        if NR is not None:
            _, _, H, W = sr.shape
            y0, x0 = max(0, H//2-256), max(0, W//2-256)
            for tag, im in (("lq", lq.to(DEV)), ("sr", sr)):
                crop = im[:, :, y0:y0+512, x0:x0+512]
                for k in stats[tag]:
                    stats[tag][k].append(float(NR[k](crop).item()))
        del lq, sr
        torch.cuda.empty_cache()
        if i % 10 == 0 or i == len(files):
            elapsed = time.time() - t0
            print(f"[{i}/{len(files)}] {elapsed/i:.2f}s/image", flush=True)

    nr_summary = None
    if NR is not None:
        nr_summary = {}
        for tag in ("lq", "sr"):
            m = {k: float(np.mean(v)) for k, v in stats[tag].items()}
            m["nr_contrib"] = nr_proxy_contrib(m)
            nr_summary[tag] = m
        print("NR summary:", nr_summary, flush=True)

    manifest = write_manifest(
        os.path.join(args.out, "candidate_manifest.json"),
        candidate_type="4k_render",
        checkpoint=os.path.abspath(args.ckpt),
        checkpoint_sha256=sha256_file(args.ckpt),
        git_commit=git_commit(os.path.dirname(__file__)),
        loader_report=report,
        alpha_mode=args.alpha_mode,
        effective_alpha_mode=report.get("alpha_mode"),
        alpha=args.alpha,
        tile=args.tile,
        overlap=args.overlap,
        colorfix=args.colorfix,
        jpeg_quality=args.quality,
        metric_names=load_metric_names(),
        nr_summary=nr_summary,
        output_hashes=output_hashes,
    )
    with open(os.path.join(args.out, "audit.json"), "w", encoding="utf-8") as f:
        json.dump({"nr": nr_summary, "elapsed_sec": time.time()-t0}, f, indent=2, ensure_ascii=False)
    print("manifest checkpoint SHA256:", manifest["checkpoint_sha256"], flush=True)


if __name__ == "__main__":
    main()
