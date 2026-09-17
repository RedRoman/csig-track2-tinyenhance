"""中壳 + 对抗蒸馏：填补质量-时延曲线上 30~120M 的空白区间。

已测两端：8.34M/1.2ms 塌（crop 2.49），340M/12.75ms 好（3.89）。
150 分的所有剧本都需要知道膝盖在哪。

两个此前未打的牌合在这里：
  1. 容量 40M/90M（unshuffle 壳放大版，仍是一步扩散形式）
  2. 对抗项 —— NTIRE ESR 系经验：小网要 GAN 才出锐度，纯蒸馏必糊。
     判别器用 PatchGAN，真样本 = 教师输出（不是 GT！学生要逼近教师的分布）。

护栏照旧：GT 兜底 + FR 势垒 + 反幻觉势垒。
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "20")

sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
sys.path.insert(0, "/cache/zihaohe")
import cv2
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from settings import ROOT
from degradation import degrade as degrade_legacy
from metrics_utils import create_metrics
from eval_utils import collect_crops, evaluate_model
from ultra_shell import Block

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16


class MidShell(nn.Module):
    """r4 unshuffle -> 三尺度 UNet 残差 -> shuffle。一步扩散形式 z-net(z)。

    r=4：latent 128x128x48，比 r8 保留更多空间分辨率（8M 壳塌掉可能与 r8 的
    64x64 过度压缩有关）。三尺度给全局感受野。
    """

    def __init__(self, r=4, base=96, blocks=(2, 3, 4)):
        super().__init__()
        self.r = r
        cin = 3 * r * r
        C = base
        self.inp = nn.Conv2d(cin, C, 3, padding=1)
        self.d1 = nn.Sequential(*[Block(C) for _ in range(blocks[0])])
        self.down1 = nn.Conv2d(C, 2 * C, 3, stride=2, padding=1)
        self.d2 = nn.Sequential(*[Block(2 * C) for _ in range(blocks[1])])
        self.down2 = nn.Conv2d(2 * C, 4 * C, 3, stride=2, padding=1)
        self.mid = nn.Sequential(*[Block(4 * C) for _ in range(blocks[2])])
        self.up2 = nn.ConvTranspose2d(4 * C, 2 * C, 4, stride=2, padding=1)
        self.u2 = nn.Sequential(*[Block(2 * C) for _ in range(blocks[1])])
        self.up1 = nn.ConvTranspose2d(2 * C, C, 4, stride=2, padding=1)
        self.u1 = nn.Sequential(*[Block(C) for _ in range(blocks[0])])
        self.out = nn.Conv2d(C, cin, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x01):
        z = F.pixel_unshuffle(x01 * 2 - 1, self.r)
        h1 = self.d1(self.inp(z))
        h2 = self.d2(self.down1(h1))
        m = self.mid(self.down2(h2))
        x2 = self.u2(self.up2(m) + h2)
        x1 = self.u1(self.up1(x2) + h1)
        zo = z - self.out(x1)
        return (F.pixel_shuffle(zo, self.r) * 0.5 + 0.5).clamp(0, 1)


class PatchD(nn.Module):
    """条件 PatchGAN：输入 (图, LQ) 拼接，让 D 判断“在这个 LQ 条件下这是不是教师风格”"""

    def __init__(self, ch=64):
        super().__init__()
        def blk(i, o, s):
            return [nn.Conv2d(i, o, 4, stride=s, padding=1),
                    nn.GroupNorm(8, o), nn.LeakyReLU(0.2, inplace=True)]
        self.net = nn.Sequential(
            nn.Conv2d(6, ch, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            *blk(ch, 2 * ch, 2), *blk(2 * ch, 4 * ch, 2), *blk(4 * ch, 4 * ch, 1),
            nn.Conv2d(4 * ch, 1, 4, padding=1))

    def forward(self, img, lq):
        return self.net(torch.cat([img, lq], 1))


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


class Wrap(nn.Module):
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
    ap.add_argument("--teacher_dir", default="/cache/zihaohe/distill_data/teacher_v3")
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--blocks", default="2,3,4")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d_lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--w_t_l1", type=float, default=1.0)
    ap.add_argument("--w_t_lpips", type=float, default=2.0)
    ap.add_argument("--w_t_dists", type=float, default=2.0)
    ap.add_argument("--w_gan", type=float, default=0.1)
    ap.add_argument("--gan_start", type=int, default=2000,
                    help="先纯蒸馏预热，再开对抗")
    ap.add_argument("--w_l1", type=float, default=0.5)
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_dists", type=float, default=1.5)
    ap.add_argument("--w_fid", type=float, default=1.5)
    ap.add_argument("--lpips_cap", type=float, default=0.26)
    ap.add_argument("--dists_cap", type=float, default=0.19)
    ap.add_argument("--w_barrier", type=float, default=3000.0)
    ap.add_argument("--w_antihall", type=float, default=20000.0)
    ap.add_argument("--antihall_cap", type=float, default=0.038)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_crops", type=int, default=12)
    ap.add_argument("--out", default=f"{ROOT}/train_out/midshell_v1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    blocks = tuple(int(x) for x in args.blocks.split(","))
    model = MidShell(args.r, args.base, blocks).to(DEV).float()
    disc = PatchD().to(DEV).float()
    wrap = Wrap(model)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"midshell 参数 {n:.1f}M (r{args.r} base{args.base} {blocks})", flush=True)

    g_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    d_opt = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, weight_decay=0.0)
    g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=args.steps,
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
    print(f"[0] score={ev['score']:.4f} PSNR={ev['metrics']['psnr']:.3f}", flush=True)

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
        # --- G ---
        t_l1 = F.l1_loss(out, te)
        t_lp = M["lpips"](out, te).mean()
        t_ds = M["dists"](out, te).mean()
        loss = args.w_t_l1 * t_l1 + args.w_t_lpips * t_lp + args.w_t_dists * t_ds
        g_lp = M["lpips"](out, gt).mean()
        g_ds = M["dists"](out, gt).mean()
        loss = loss + args.w_l1 * F.l1_loss(out, gt) + args.w_lpips * g_lp + args.w_dists * g_ds
        loss = loss + args.w_fid * F.l1_loss(F.avg_pool2d(out, 16), F.avg_pool2d(lq, 16))
        bar = F.relu(g_lp - args.lpips_cap) ** 2 + F.relu(g_ds - args.dists_cap) ** 2
        loss = loss + args.w_barrier * bar
        lo = F.avg_pool2d(out, 16); ll = F.avg_pool2d(lq, 16)
        dev_ = (lo - ll).abs().mean(1).flatten(1)
        kk = max(1, int(dev_.shape[1] * 0.001))
        top = torch.topk(dev_, kk, dim=1).values.mean()
        loss = loss + args.w_antihall * F.relu(top - args.antihall_cap) ** 2
        gan_on = step >= args.gan_start
        if gan_on:
            loss = loss + args.w_gan * F.softplus(-disc(out, lq)).mean()

        g_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        g_opt.step(); g_sched.step()

        # --- D（教师输出为真样本）---
        if gan_on:
            d_loss = (F.softplus(disc(out.detach(), lq)).mean()
                      + F.softplus(-disc(te, lq)).mean())
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()
        else:
            d_loss = torch.tensor(0.0)

        if step % 100 == 0:
            print(f"[{step}/{args.steps}] G={loss.item():.4f} D={d_loss.item():.4f} "
                  f"tLPIPS={t_lp.item():.4f} gLPIPS={g_lp.item():.4f} lf_top={top.item():.4f} "
                  f"gan={'on' if gan_on else 'off'} {(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            ev = evaluate_model(wrap, val_items, M, device=DEV, dtype=DT, robust_beta=0.15)
            ev.pop("rows", None)
            hist.append({"step": step, "score": float(ev["score"]),
                         **{k: float(v) for k, v in ev["metrics"].items()}})
            flag = ""
            if ev["robust_score"] > best:
                best = ev["robust_score"]; flag = " <- best"
                torch.save({"net": model.state_dict(),
                            "cfg": dict(r=args.r, base=args.base, blocks=blocks),
                            "step": step, "args": vars(args)},
                           os.path.join(args.out, "best.pth"))
            print(f"[val {step}] score={ev['score']:.4f} robust={ev['robust_score']:.4f} "
                  f"PSNR={ev['metrics']['psnr']:.3f}{flag}", flush=True)
            json.dump(hist, open(os.path.join(args.out, "log.json"), "w"), indent=2)
    print(f"done. best robust={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
