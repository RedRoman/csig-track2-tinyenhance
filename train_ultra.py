"""子毫秒壳蒸馏：teacher_koniq(86.5ms, Score 4.08) -> ShuffleNet1Step(0.8-1.2ms)。

配方 = distill_v1/v2 的（Score 保留率 100% 已两次验证），改动：
  - 学生换 pixel_unshuffle 无损壳（无 TAESD，无信息瓶颈，FR 地板完美）
  - 没有共享 latent，蒸馏项全在像素域
  - 护栏照旧：FR 势垒 + 反幻觉势垒 + 小权重 IQA（koniq 口径）
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "20")

sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
sys.path.insert(0, "/cache/zihaohe")
import cv2
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from settings import ROOT
from degradation import degrade as degrade_legacy
from metrics_utils import create_metrics
from eval_utils import collect_crops, evaluate_model
from ultra_shell import ShuffleNet1Step

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16


def crop_seed(idx):
    return (idx * 104729 + 7919) % (2 ** 31)


class DistillDataset(Dataset):
    def __init__(self, paths, teacher_dir, size=512):
        self.size = int(size)
        self.teacher_dir = teacher_dir
        self.items = [(i, p) for i, p in enumerate(paths)
                      if os.path.exists(os.path.join(teacher_dir, f"{i:06d}.png"))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, path = self.items[k]
        rng = np.random.RandomState(crop_seed(i))
        gt = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        h, w, _ = gt.shape
        s = self.size
        if h > s or w > s:
            y = rng.randint(0, max(h - s, 0) + 1); x = rng.randint(0, max(w - s, 0) + 1)
            gt = gt[y:y + s, x:x + s]
        if gt.shape[0] != s or gt.shape[1] != s:
            gt = cv2.resize(gt, (s, s), interpolation=cv2.INTER_CUBIC)
        if rng.random() < 0.5:
            gt = gt[:, ::-1]
        k4 = int(rng.randint(4))
        if k4:
            gt = np.rot90(gt, k4)
        gt = np.ascontiguousarray(gt)
        lq = degrade_legacy(gt, rng)
        te = np.asarray(Image.open(os.path.join(self.teacher_dir, f"{i:06d}.png")
                                   ).convert("RGB"), dtype=np.uint8)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.).permute(2, 0, 1)
        return to_t(lq), to_t(gt), to_t(te)


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.alpha_gate = None

    def forward(self, x):
        return self.m(x)

    def forward_train(self, x):
        out = self.m(x)
        return out, torch.ones(1, 1, 1, 1, device=out.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", default="/cache/zihaohe/distill_data/teacher_koniq")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--nblocks", type=int, default=6)
    ap.add_argument("--down", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--w_t_l1", type=float, default=1.0)
    ap.add_argument("--w_t_lpips", type=float, default=2.0)
    ap.add_argument("--w_t_dists", type=float, default=2.0)
    ap.add_argument("--w_l1", type=float, default=0.5)
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_dists", type=float, default=1.5)
    ap.add_argument("--w_fid", type=float, default=1.5)
    ap.add_argument("--lpips_cap", type=float, default=0.26)
    ap.add_argument("--dists_cap", type=float, default=0.19)
    ap.add_argument("--w_barrier", type=float, default=3000.0)
    ap.add_argument("--w_antihall", type=float, default=20000.0)
    ap.add_argument("--antihall_cap", type=float, default=0.038)
    ap.add_argument("--w_iqa", type=float, default=0.5)
    ap.add_argument("--iqa_batch", type=int, default=1)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_crops", type=int, default=12)
    ap.add_argument("--out", default=f"{ROOT}/train_out/ultra_v1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = ShuffleNet1Step(r=args.r, base=args.base, nblocks=args.nblocks,
                            down=args.down).to(DEV).float()
    wrap = Wrap(model)
    params = list(model.parameters())
    print(f"ultra net 参数 {sum(p.numel() for p in params)/1e6:.2f}M "
          f"(r{args.r} base{args.base} x{args.nblocks} down={args.down})", flush=True)
    opt = torch.optim.AdamW([{"params": params, "lr": args.lr}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.05)
    M = create_metrics(DEV, as_loss=True, include_psnr=True)

    paths = sorted(glob.glob(os.path.join(ROOT, "train_data", "lsdir512", "*.png")))
    ds = DistillDataset(paths, args.teacher_dir)
    print(f"蒸馏样本 {len(ds)} 对", flush=True)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)

    val_items = collect_crops(args.val_crops)
    ev = evaluate_model(wrap, val_items, M, device=DEV, dtype=DT, robust_beta=0.15)
    ev.pop("rows", None)
    print(f"[0] score={ev['score']:.4f} PSNR={ev['metrics']['psnr']:.3f} (恒等起点)", flush=True)

    best, hist, step, t0 = -1e9, [], 0, time.time()
    it = iter(dl)
    while step < args.steps:
        try:
            lq, gt, te = next(it)
        except StopIteration:
            it = iter(dl); continue
        step += 1
        lq = lq.to(DEV, non_blocking=True).float()
        gt = gt.to(DEV, non_blocking=True).float()
        te = te.to(DEV, non_blocking=True).float()
        out = model(lq)

        t_l1 = F.l1_loss(out, te)
        t_lp = M["lpips"](out, te).mean()
        t_ds = M["dists"](out, te).mean()
        loss = args.w_t_l1 * t_l1 + args.w_t_lpips * t_lp + args.w_t_dists * t_ds
        g_lp = M["lpips"](out, gt).mean()
        g_ds = M["dists"](out, gt).mean()
        loss = loss + args.w_l1 * F.l1_loss(out, gt) + args.w_lpips * g_lp + args.w_dists * g_ds
        loss = loss + args.w_fid * F.l1_loss(F.avg_pool2d(out, 16), F.avg_pool2d(lq, 16))
        if args.w_barrier > 0:
            bar = F.relu(g_lp - args.lpips_cap) ** 2 + F.relu(g_ds - args.dists_cap) ** 2
            loss = loss + args.w_barrier * bar
        if args.w_antihall > 0:
            lo = F.avg_pool2d(out, 16); ll = F.avg_pool2d(lq, 16)
            dev_ = (lo - ll).abs().mean(1).flatten(1)
            kk = max(1, int(dev_.shape[1] * 0.001))
            top = torch.topk(dev_, kk, dim=1).values.mean()
            loss = loss + args.w_antihall * F.relu(top - args.antihall_cap) ** 2
        if args.w_iqa > 0:
            iq = out[:min(args.iqa_batch, out.shape[0])]
            which = step % 3
            if which == 0:
                r = 3.0 * M["clipiqa"](iq).mean()
            elif which == 1:
                r = 3.0 * M["maniqa"](iq).mean()
            else:
                r = 3.0 * (M["musiq"](iq).mean() / 100.0)
            loss = loss - args.w_iqa * r

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if step % 100 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} tL1={t_l1.item():.4f} "
                  f"tLPIPS={t_lp.item():.4f} gLPIPS={g_lp.item():.4f} "
                  f"gDISTS={g_ds.item():.4f} {(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            ev = evaluate_model(wrap, val_items, M, device=DEV, dtype=DT, robust_beta=0.15)
            ev.pop("rows", None)
            hist.append({"step": step, "score": float(ev["score"]),
                         **{k: float(v) for k, v in ev["metrics"].items()}})
            flag = ""
            if ev["robust_score"] > best:
                best = ev["robust_score"]; flag = " <- best"
                torch.save({"net": model.state_dict(),
                            "cfg": dict(r=args.r, base=args.base,
                                        nblocks=args.nblocks, down=args.down),
                            "step": step, "args": vars(args)},
                           os.path.join(args.out, "best.pth"))
            print(f"[val {step}] score={ev['score']:.4f} robust={ev['robust_score']:.4f} "
                  f"PSNR={ev['metrics']['psnr']:.3f}{flag}", flush=True)
            json.dump(hist, open(os.path.join(args.out, "log.json"), "w"), indent=2)
    print(f"done. best robust={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
