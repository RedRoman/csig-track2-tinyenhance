"""Exact weight-space soup between two TinySR checkpoints.

Important: averaging LoRA A/B factors is NOT equivalent to averaging the merged
models. This script first merges each adapter into the Transformer, then performs
W(lambda)=(1-lambda)W_A+lambda W_B in dense weight space.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import torch

from settings import ROOT
from tinysr_runtime import build_model, read_checkpoint

DEV = torch.device("cuda")
DT = torch.float16


def merged_state(path):
    model, report = build_model(path, device=DEV, dtype=DT, merge_lora=True,
                                alpha_mode="fixed", fixed_alpha=1.0)
    state = {k: v.detach().cpu() for k, v in model.tr.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return state, report


def mix_state(a, b, lam):
    out = {}
    for k, va in a.items():
        vb = b.get(k)
        if vb is None or tuple(va.shape) != tuple(vb.shape) or not torch.is_floating_point(va):
            out[k] = va.clone()
        else:
            # Interpolate in fp32 on CPU then store fp16 to control checkpoint size.
            out[k] = ((1.0 - lam) * va.float() + lam * vb.float()).to(va.dtype)
    return out


def mix_optional_state(sa, sb, key, lam):
    a, b = sa.get(key), sb.get(key)
    if not a and not b:
        return None
    if not a:
        return b
    if not b:
        return a
    out = {}
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None and tuple(va.shape) == tuple(vb.shape) and torch.is_floating_point(va):
            out[k] = ((1-lam)*va.float() + lam*vb.float()).to(va.dtype)
        else:
            out[k] = va
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="conservative checkpoint, e.g. run11")
    ap.add_argument("--b", required=True, help="aggressive checkpoint, e.g. run12")
    ap.add_argument("--lambdas", default="0.0,0.2,0.4,0.6,0.8,1.0,1.1")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "train_out", "weight_soup"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("merging A ...", flush=True)
    A, ra = merged_state(args.a)
    print("A:", ra)
    print("merging B ...", flush=True)
    B, rb = merged_state(args.b)
    print("B:", rb)
    raw_a, raw_b = read_checkpoint(args.a), read_checkpoint(args.b)

    index = []
    for lam in [float(x) for x in args.lambdas.split(",")]:
        out = os.path.join(args.out_dir, f"soup_{lam:.2f}.pth")
        payload = {
            "format_version": 2,
            "step": -1,
            "merged_tr": mix_state(A, B, lam),
            "vae_decoder": mix_optional_state(raw_a, raw_b, "vae_decoder", lam),
            "alpha_gate": None,
            "alpha_config": None,
            "soup": {"a": args.a, "b": args.b, "lambda": lam},
        }
        torch.save(payload, out)
        size_mb = os.path.getsize(out) / 1e6
        index.append({"lambda": lam, "path": out, "size_mb": size_mb})
        print(f"lambda={lam:.2f} -> {out} ({size_mb:.1f} MB)", flush=True)
        del payload
        gc.collect()

    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
