"""CSIG track-2 TinySR v2 training.

Recommended first ablation:
  python code/train_tinysr.py --init_ckpt <run12/best.pth> \
      --adaptive_alpha --alpha_min 0.72 --alpha_max 1.10 \
      --w_iqa 1.0 --w_maniqa 1.0 --w_alpha_prior 0.02 \
      --steps 1800 --lr 1e-5 --out <.../v2_alpha>

The script keeps the old LoRA path available, while adding:
  * unified CLIP-IQA + MANIQA + MUSIQ optimization;
  * optional severity-adaptive latent residual strength alpha(x);
  * robust checkpoint selection across cases/tier variance;
  * optional VAE decoder and last-N DiT dense fine-tuning (zero inference ops);
  * optional narrow ISP-mixture degradation.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import time
import math
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from settings import ROOT, VAL
from degradation import degrade as degrade_legacy, degrade_v2
from metrics_utils import create_metrics, proxy_score, load_metric_names
from tinysr_runtime import build_model, checkpoint_payload

Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")
DT = torch.float16


def load_img(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


class PairDataset(Dataset):
    def __init__(self, paths, size=512, repeat=1, degradation="legacy"):
        self.paths = paths * repeat
        self.size = int(size)
        self.degradation = degradation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        seed = (int(torch.initial_seed()) % 100003 * 7919 + i * 104729) % (2 ** 31)
        rng = np.random.RandomState(seed)
        gt = np.asarray(Image.open(self.paths[i]).convert("RGB"), dtype=np.uint8)
        h, w, _ = gt.shape
        s = self.size
        if h > s or w > s:
            y = rng.randint(0, max(h - s, 0) + 1)
            x = rng.randint(0, max(w - s, 0) + 1)
            gt = gt[y:y + s, x:x + s]
        if gt.shape[0] != s or gt.shape[1] != s:
            gt = cv2.resize(gt, (s, s), interpolation=cv2.INTER_CUBIC)
        if rng.random() < 0.5:
            gt = gt[:, ::-1]
        k = int(rng.randint(4))
        if k:
            gt = np.rot90(gt, k)
        gt = np.ascontiguousarray(gt)

        operator_severity = None
        if self.degradation == "v2":
            lq, meta = degrade_v2(gt, rng, return_meta=True)
            operator_severity = float(meta["severity"])
        else:
            # Preserve the original calibrated chain for clean ablations.
            lq = degrade_legacy(gt, rng)

        # Weak severity supervision is computed from the synthetic pair itself.
        # Mapping roughly covers the three real validation PSNR regimes (~21/28/33 dB).
        mse = float(np.mean((lq.astype(np.float32) - gt.astype(np.float32)) ** 2))
        psnr = 10.0 * np.log10((255.0 ** 2) / max(mse, 1e-8))
        pair_severity = float(np.clip((34.0 - psnr) / 14.0, 0.0, 1.0))
        severity = pair_severity if operator_severity is None else 0.5 * (pair_severity + operator_severity)

        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        return to_t(lq), to_t(gt), torch.tensor(severity, dtype=torch.float32)


def build_val_set(n_per=12):
    pairs = [("case1_lq.jpg", "case1_gt.png", 101, "case1"),
             ("case2_lq.jpg", "case2_gt.png", 202, "case2"),
             ("case3_lq.jpg", "case3_gt.jpg", 303, "case3")]
    items = []
    for lqn, gtn, seed, case in pairs:
        lq_f, gt_f = load_img(os.path.join(VAL, lqn)), load_img(os.path.join(VAL, gtn))
        if lq_f.shape != gt_f.shape:
            continue
        _, _, H, W = gt_f.shape
        rng = np.random.RandomState(seed)
        got, tries = 0, 0
        while got < n_per and tries < n_per * 20:
            tries += 1
            y = rng.randint(0, max(H - 512, 1))
            x = rng.randint(0, max(W - 512, 1))
            gt = gt_f[:, :, y:y + 512, x:x + 512]
            if gt.shape[-2:] != (512, 512) or gt.std().item() < 0.035:
                continue
            lq = lq_f[:, :, y:y + 512, x:x + 512]
            psnr_lq = -10.0 * np.log10(F.mse_loss(lq, gt).item() + 1e-12)
            items.append({"lq": lq.clone(), "gt": gt.clone(), "psnr_lq": psnr_lq,
                          "case": case})
            got += 1
    ps = sorted(i["psnr_lq"] for i in items)
    lo, hi = ps[len(ps) // 3], ps[2 * len(ps) // 3]
    for it in items:
        it["tier"] = "heavy" if it["psnr_lq"] < lo else ("light" if it["psnr_lq"] > hi else "medium")
    return items


@torch.no_grad()
def evaluate(model, items, metrics, robust_beta=0.15):
    was_train = model.training
    model.eval()
    all_scores, psnr_out = [], []
    case_scores, tier_scores, alphas = defaultdict(list), defaultdict(list), []
    for it in items:
        lq = it["lq"].to(DEV, dtype=DT)
        gt = it["gt"].to(DEV)
        with torch.autocast(device_type="cuda", dtype=DT):
            out, alpha = model.forward_train(lq)
        out = out.float()
        vals = {}
        for k, fn in metrics.items():
            if k == "psnr":
                continue
            v = fn(out, gt) if k in ("lpips", "dists") else fn(out)
            vals[k] = float(v.item())
        sc = proxy_score(vals)
        all_scores.append(sc)
        case_scores[it["case"]].append(sc)
        tier_scores[it["tier"]].append(sc)
        psnr_out.append(-10.0 * np.log10(F.mse_loss(out, gt).item() + 1e-12))
        alphas.append(float(alpha.float().mean().item()))
        del lq, gt, out
        torch.cuda.empty_cache()
    if was_train:
        model.train()
    case_mean = {k: float(np.mean(v)) for k, v in case_scores.items()}
    tier_mean = {k: float(np.mean(v)) for k, v in tier_scores.items()}
    raw = float(np.mean(all_scores))
    robust = raw - float(robust_beta) * float(np.std(list(case_mean.values())))
    return {
        "score": raw,
        "robust_score": robust,
        "psnr": float(np.mean(psnr_out)),
        "case": case_mean,
        "tier": tier_mean,
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
    }


def select_dense_params(model, last_n):
    """Unfreeze the last N DiT blocks without changing inference structure."""
    if last_n <= 0:
        return [], []
    blocks = list(model.tr.transformer_blocks)
    chosen = blocks[-min(last_n, len(blocks)):]
    ids = {id(p) for b in chosen for p in b.parameters()}
    params, names = [], []
    for name, p in model.tr.named_parameters():
        if id(p) in ids and "lora_" not in name:
            p.requires_grad_(True)
            p.data = p.data.float()  # fp32 master weight; autocast keeps forward efficient
            params.append(p)
            names.append(name)
    return params, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--dense_lr", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no_grad_scaler", action="store_true")
    ap.add_argument("--val_every", type=int, default=150)
    ap.add_argument("--val_crops", type=int, default=12)
    ap.add_argument("--robust_beta", type=float, default=0.15)
    ap.add_argument("--select_by", choices=["score", "robust_score"], default="robust_score")

    ap.add_argument("--w_l1", type=float, default=0.5)
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_dists", type=float, default=1.5)
    ap.add_argument("--w_fid", type=float, default=1.5)
    ap.add_argument("--w_iqa", type=float, default=1.0)
    ap.add_argument("--w_clipiqa", type=float, default=1.0)
    ap.add_argument("--w_maniqa", type=float, default=1.0)
    ap.add_argument("--w_musiq", type=float, default=1.0)
    ap.add_argument("--iqa_batch", type=int, default=2,
                    help="IQA loss uses first N batch items to cap memory")
    ap.add_argument("--iqa_mode", choices=["roundrobin", "all"], default="roundrobin",
                    help="roundrobin gives an unbiased 3-metric objective with much lower activation memory")

    ap.add_argument("--adaptive_alpha", action="store_true")
    ap.add_argument("--fixed_alpha", type=float, default=1.0)
    ap.add_argument("--alpha_min", type=float, default=0.72)
    ap.add_argument("--alpha_max", type=float, default=1.10)
    ap.add_argument("--alpha_hidden", type=int, default=32)
    ap.add_argument("--w_alpha_prior", type=float, default=0.02,
                    help="weak synthetic severity ordering; set 0 for fully unsupervised gate")
    ap.add_argument("--alpha_light", type=float, default=0.82)
    ap.add_argument("--alpha_heavy", type=float, default=1.06)

    ap.add_argument("--train_vae_decoder", action="store_true")
    ap.add_argument("--freeze_lora", action="store_true",
                    help="train only new zero-overhead modules (e.g. alpha gate/decoder), useful for clean ablations")
    ap.add_argument("--unfreeze_last_n", type=int, default=0)
    ap.add_argument("--degradation", choices=["legacy", "v2"], default="legacy")
    ap.add_argument("--extra_data", type=str, default=None)
    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "train_out", "v2_alpha"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Metric config:", load_metric_names(), flush=True)
    paths = (sorted(glob.glob(os.path.join(ROOT, "AdcSR/testset/DIV2K-Val/HR/*.png")))
             + sorted(glob.glob(os.path.join(ROOT, "AdcSR/testset/DRealSR/HR/*.png")))
             + sorted(glob.glob(os.path.join(ROOT, "AdcSR/testset/RealSR/HR/*.png"))))
    if args.extra_data:
        for d in args.extra_data.split(","):
            paths += sorted(glob.glob(os.path.join(d.strip(), "*.png")))
    rep = 20 if len(paths) < 8000 else max(1, 60000 // max(len(paths), 1))
    print(f"train images={len(paths)} repeat={rep} degradation={args.degradation}", flush=True)
    dl = DataLoader(PairDataset(paths, repeat=rep, degradation=args.degradation),
                    batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)

    model, report = build_model(
        args.init_ckpt, device=DEV, dtype=DT, merge_lora=False,
        alpha_mode="adaptive" if args.adaptive_alpha else "fixed",
        fixed_alpha=args.fixed_alpha, adaptive_alpha=args.adaptive_alpha,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max, alpha_hidden=args.alpha_hidden,
        init_alpha=args.fixed_alpha)
    print("load report:", report, flush=True)

    # Freeze first, then explicitly open only the zero-inference-overhead adaptation groups.
    for p in model.parameters():
        p.requires_grad_(False)
    main_params = []
    for name, p in model.tr.named_parameters():
        if "lora_" in name and not args.freeze_lora:
            p.requires_grad_(True)
            p.data = p.data.float()
            main_params.append(p)
    if model.alpha_gate is not None:
        for p in model.alpha_gate.parameters():
            p.requires_grad_(True)
            p.data = p.data.float()
            main_params.append(p)
    if args.train_vae_decoder:
        for name, p in model.vae.named_parameters():
            if name.startswith("decoder"):
                p.requires_grad_(True)
                p.data = p.data.float()  # stable AdamW master weights; forward is under autocast
                main_params.append(p)
    dense_params, dense_names = select_dense_params(model, args.unfreeze_last_n)

    if not main_params and not dense_params:
        raise ValueError("no trainable parameters: enable LoRA, adaptive alpha, decoder, or dense FT")
    groups = []
    if main_params:
        groups.append({"params": main_params, "lr": args.lr})
    if dense_params:
        groups.append({"params": dense_params, "lr": args.dense_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    # Multiplicative cosine keeps the dense/main LR ratio intact.
    def lr_factor(s):
        t = min(max(float(s) / max(args.steps, 1), 0.0), 1.0)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=not args.no_grad_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=not args.no_grad_scaler)
    print(f"trainable main={sum(p.numel() for p in main_params)/1e6:.2f}M "
          f"dense={sum(p.numel() for p in dense_params)/1e6:.2f}M", flush=True)

    # One shared metric set: as_loss=True only enables gradients; under no_grad it
    # is also the validation scorer. Avoid duplicating MANIQA/MUSIQ/CLIP weights on GPU.
    loss_metrics = create_metrics(DEV, as_loss=True)
    val_metrics = loss_metrics
    val_items = build_val_set(args.val_crops)
    print(f"validation crops={len(val_items)} selection={args.select_by} beta={args.robust_beta}", flush=True)

    ev0 = evaluate(model, val_items, val_metrics, args.robust_beta)
    print(f"[0] score={ev0['score']:.4f} robust={ev0['robust_score']:.4f} "
          f"PSNR={ev0['psnr']:.3f} alpha={ev0['alpha_mean']:.3f}±{ev0['alpha_std']:.3f} "
          f"case={ev0['case']} tier={ev0['tier']}", flush=True)

    best = -1e9
    step = 0
    t0 = time.time()
    log = []
    model.train()
    while step < args.steps:
        for lq, gt, severity in dl:
            if step >= args.steps:
                break
            lq = lq.to(DEV, non_blocking=True)
            gt = gt.to(DEV, non_blocking=True)
            severity = severity.to(DEV, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=DT):
                out, alpha = model.forward_train(lq)
                out = out.float()
                l1 = F.l1_loss(out, gt)
                lp = loss_metrics["lpips"](out, gt).mean()
                ds = loss_metrics["dists"](out, gt).mean()
                loss = args.w_l1 * l1 + args.w_lpips * lp + args.w_dists * ds

                fid = F.l1_loss(F.avg_pool2d(out, 16), F.avg_pool2d(lq, 16))
                loss = loss + args.w_fid * fid

                n_iqa = min(args.iqa_batch, out.shape[0])
                iq = out[:n_iqa]
                # All three are higher-better. In round-robin mode, multiplying
                # the sampled term by 3 makes the gradient an unbiased estimate
                # of the full three-metric sum while avoiding three large graphs.
                ci = torch.tensor(float("nan"), device=DEV)
                ma = torch.tensor(float("nan"), device=DEV)
                mu = torch.tensor(float("nan"), device=DEV)
                if args.iqa_mode == "all":
                    ci = loss_metrics["clipiqa"](iq).mean()
                    ma = loss_metrics["maniqa"](iq).mean()
                    mu = loss_metrics["musiq"](iq).mean() / 100.0
                    iqa_reward = args.w_clipiqa * ci + args.w_maniqa * ma + args.w_musiq * mu
                else:
                    which = step % 3
                    if which == 0:
                        ci = loss_metrics["clipiqa"](iq).mean()
                        iqa_reward = 3.0 * args.w_clipiqa * ci
                    elif which == 1:
                        ma = loss_metrics["maniqa"](iq).mean()
                        iqa_reward = 3.0 * args.w_maniqa * ma
                    else:
                        mu = loss_metrics["musiq"](iq).mean() / 100.0
                        iqa_reward = 3.0 * args.w_musiq * mu
                loss = loss - args.w_iqa * iqa_reward

                alpha_prior = torch.tensor(0.0, device=DEV)
                if model.alpha_gate is not None and args.w_alpha_prior > 0:
                    target = args.alpha_light + severity[:, None, None, None] * (args.alpha_heavy - args.alpha_light)
                    alpha_prior = F.smooth_l1_loss(alpha.float(), target.float())
                    loss = loss + args.w_alpha_prior * alpha_prior

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            if step % 50 == 0:
                print(f"[{step}/{args.steps}] loss={loss.item():.4f} l1={l1.item():.4f} "
                      f"lp={lp.item():.4f} dists={ds.item():.4f} fid={fid.item():.4f} "
                      f"IQA={iqa_reward.item():.4f} clip={ci.item():.4f} maniqa={ma.item():.4f} musiq={mu.item():.4f} "
                      f"alpha={alpha.float().mean().item():.3f} prior={alpha_prior.item():.4f} "
                      f"{(time.time()-t0)/step:.2f}s/it", flush=True)

            if step % args.val_every == 0:
                ev = evaluate(model, val_items, val_metrics, args.robust_beta)
                key = ev[args.select_by]
                row = {"step": step, **ev, "selection": key}
                log.append(row)
                flag = ""
                if key > best:
                    best = key
                    payload = checkpoint_payload(
                        model, step, train_vae_decoder=args.train_vae_decoder,
                        dense_names=dense_names, args_dict=vars(args))
                    payload["val"] = ev
                    torch.save(payload, os.path.join(args.out, "best.pth"))
                    flag = " <- best"
                print(f"[val {step}] score={ev['score']:.4f} robust={ev['robust_score']:.4f} "
                      f"PSNR={ev['psnr']:.3f} alpha={ev['alpha_mean']:.3f}±{ev['alpha_std']:.3f} "
                      f"case={ev['case']} tier={ev['tier']}{flag}", flush=True)
                with open(os.path.join(args.out, "log.json"), "w", encoding="utf-8") as f:
                    json.dump(log, f, indent=2, ensure_ascii=False)
                model.train()

    torch.save(checkpoint_payload(model, step, train_vae_decoder=args.train_vae_decoder,
                                  dense_names=dense_names, args_dict=vars(args)),
               os.path.join(args.out, "final.pth"))
    print(f"done. best {args.select_by}={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
