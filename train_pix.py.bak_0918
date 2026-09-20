"""moonshot:像素空间 UNet 从零蒸馏 v8(教师=v8_teacher 20k)。损失配方沿用 train_distill(v8 档)。"""
import os, sys, glob, json, time, math, argparse, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code"); sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from settings import ROOT
from metrics_utils import create_metrics
from eval_utils import collect_crops, evaluate_model
from train_distill import DistillDataset, PatchD
from pix_model import PixUNet

DEV, DT = torch.device("cuda"), torch.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", default="/cache/zihaohe/distill_data/v8_teacher")
    ap.add_argument("--data_mix", type=float, default=0.5)
    ap.add_argument("--arch", default="128,256,512"); ap.add_argument("--nb", type=int, default=3)
    ap.add_argument("--ps", type=int, default=4); ap.add_argument("--tdim", type=int, default=128)
    ap.add_argument("--init_ckpt", default="")
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--min_lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--w_t_l1", type=float, default=1.0); ap.add_argument("--w_t_lpips", type=float, default=2.0)
    ap.add_argument("--w_t_dists", type=float, default=2.0)
    ap.add_argument("--w_t_hf", type=float, default=0.0, help="对教师的高频 L1(out-blur(out) vs te-blur(te)),逼学生复刻教师纹理而非取均值")
    ap.add_argument("--w_l1", type=float, default=0.5); ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_dists", type=float, default=1.5); ap.add_argument("--w_fid", type=float, default=1.5)
    ap.add_argument("--lpips_cap", type=float, default=0.26); ap.add_argument("--dists_cap", type=float, default=0.19)
    ap.add_argument("--w_barrier", type=float, default=3000.0)
    ap.add_argument("--w_iqa", type=float, default=2.0); ap.add_argument("--iqa_start", type=int, default=3000)
    ap.add_argument("--w_maniqa", type=float, default=3.0); ap.add_argument("--w_clipiqa", type=float, default=2.0)
    ap.add_argument("--w_musiq", type=float, default=0.3); ap.add_argument("--w_niqe", type=float, default=1.2)
    ap.add_argument("--w_gan", type=float, default=0.05); ap.add_argument("--gan_start", type=int, default=5000)
    ap.add_argument("--d_lr", type=float, default=1e-4); ap.add_argument("--w_hf_over", type=float, default=1500.0)
    ap.add_argument("--iqa_batch", type=int, default=2); ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--val_every", type=int, default=1000); ap.add_argument("--val_crops", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat_dir", default="", help="教师特征目录(v8_feat_b7),启用特征蒸馏")
    ap.add_argument("--iqa_eot", type=int, default=0, help=">0: IQA 对随机变换(平移/翻转/缩放)后的输出计算,取 N 次均值——对抗像素级 metric-gaming 纹理")
    ap.add_argument("--w_feat", type=float, default=1.0)
    ap.add_argument("--feat_loss", default="cos", choices=["cos", "mse"])
    ap.add_argument("--iqa_cap_auto", action="store_true",
                    help="启动时用 init 模型在 32 张训练 crop 上测 maniqa/clipiqa/musiq 均值作为上限;IQA 项改为 hinge(达到上限即无梯度),把优化空间让给 FR")
    ap.add_argument("--iqa_cap_margin", type=float, default=0.0, help="上限再加的余量(可为负=主动压低)")
    ap.add_argument("--out", default=f"{ROOT}/train_out/pix_v1")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    chs = tuple(int(c) for c in args.arch.split(","))
    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model = PixUNet(**sd["cfg"]); model.load_state_dict(sd["state"]); print("init from", args.init_ckpt, flush=True)
    else:
        model = PixUNet(chs=chs, nb=args.nb, tdim=args.tdim, ps=args.ps)
    model = model.to(DEV).float().train()
    params = [p for p in model.parameters()]
    proj = None  # 占位,特征蒸馏时在数据集构建后赋值并并入优化器
    print(f"PixUNet chs={chs} nb={args.nb} ps={args.ps} 参数 {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)
    opt = None  # 延后到 proj 确定后创建
    disc, d_opt = None, None
    if args.w_gan > 0:
        disc = PatchD().to(DEV); d_opt = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    M = create_metrics(DEV, as_loss=True, include_psnr=True)
    paths = sorted(glob.glob(os.path.join(ROOT, "train_data", "lsdir512", "*.png")))
    ds = DistillDataset(paths, args.teacher_dir, mix=args.data_mix)
    if args.feat_dir:
        class FeatDS(torch.utils.data.Dataset):
            def __init__(s, base, fdir):
                s.base = base; s.fdir = fdir
                s.items = [(i, p) for (i, p) in base.items if os.path.exists(os.path.join(fdir, f"{i:06d}.pt"))]
                base.items = s.items
            def __len__(s): return len(s.items)
            def __getitem__(s, k):
                lq, gt, te = s.base[k]
                f = torch.load(os.path.join(s.fdir, f"{s.items[k][0]:06d}.pt"), map_location="cpu", weights_only=True)
                return lq, gt, te, f.float().permute(2, 0, 1)   # [256,32,32]
        ds = FeatDS(ds, args.feat_dir)
        pca = torch.load(os.path.join(args.feat_dir, "pca.pt"), map_location="cpu", weights_only=True)
        fk = pca["V"].shape[1]
        proj = torch.nn.Sequential(torch.nn.Conv2d(int(model.cfg["chs"][-1]), fk, 1), torch.nn.GELU(), torch.nn.Conv2d(fk, fk, 1)).to(DEV)
        print(f"特征蒸馏: dir={args.feat_dir} block={int(pca['block'])} k={fk} loss={args.feat_loss} w={args.w_feat}", flush=True)
    else:
        proj = None
    print(f"蒸馏样本 {len(ds)} 对", flush=True)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.workers > 0)
    if proj is not None:
        params = params + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    ema = [p.detach().clone() for p in params]
    val_items = collect_crops(args.val_crops)
    from hf_regularization import gt_aware_hf_overshoot_loss
    caps = None
    if args.iqa_cap_auto:
        model.eval(); vals = {"maniqa": [], "clipiqa": [], "musiq": []}
        with torch.no_grad():
            it0 = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=2))
            for _ in range(24):
                lq0, _, _ = next(it0)
                with torch.autocast(device_type="cuda", dtype=DT):
                    o0, _ = model.forward_train(lq0.to(DEV))
                o0 = o0.float()
                for k in vals: vals[k].append(float(M[k](o0).mean())); torch.cuda.empty_cache()
        caps = {k: float(np.mean(v)) + args.iqa_cap_margin * (100.0 if k == "musiq" else 1.0) for k, v in vals.items()}
        model.train(); print(f"IQA hinge 上限(init 模型 32 crop 均值+margin): {caps}", flush=True)

    def lr_at(step):
        if step < args.warmup: return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))

    def swap_ema(to_ema):
        nonlocal backup
        with torch.no_grad():
            if to_ema:
                backup = [p.detach().clone() for p in params]
                for p, e in zip(params, ema): p.copy_(e)
            else:
                for p, b in zip(params, backup): p.copy_(b)
    backup = None
    best, hist, step, t0 = -1e9, [], 0, time.time(); it = iter(dl)
    while step < args.steps:
        try: batch = next(it)
        except StopIteration: it = iter(dl); continue
        lq, gt, te = batch[:3]; tf = batch[3].to(DEV, non_blocking=True) if len(batch) > 3 else None
        step += 1
        for g in opt.param_groups: g["lr"] = lr_at(step)
        lq = lq.to(DEV, non_blocking=True); gt = gt.to(DEV, non_blocking=True); te = te.to(DEV, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=DT):
            out, _, mid = model.forward_train(lq, return_feat=True)
        out = out.float(); gt = gt.float(); lq = lq.float(); te = te.float()
        f_loss = torch.tensor(0.0, device=DEV)
        if proj is not None and tf is not None:
            sf = proj(F.adaptive_avg_pool2d(mid.float(), tf.shape[-2:]))
            if args.feat_loss == "cos":
                f_loss = (1 - F.cosine_similarity(sf, tf, dim=1)).mean()
            else:
                f_loss = F.mse_loss(sf, tf / (tf.flatten(1).std(1)[:, None, None, None] + 1e-6))
        t_l1 = F.l1_loss(out, te); t_lp = M["lpips"](out, te).mean(); t_ds = M["dists"](out, te).mean()
        loss = args.w_t_l1 * t_l1 + args.w_t_lpips * t_lp + args.w_t_dists * t_ds + args.w_feat * f_loss
        if args.w_t_hf > 0:
            hp = lambda z: z - F.avg_pool2d(F.pad(z, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
            loss = loss + args.w_t_hf * F.l1_loss(hp(out), hp(te))
        g_l1 = F.l1_loss(out, gt); g_lp = M["lpips"](out, gt).mean(); g_ds = M["dists"](out, gt).mean()
        loss = loss + args.w_l1 * g_l1 + args.w_lpips * g_lp + args.w_dists * g_ds
        loss = loss + args.w_fid * F.l1_loss(F.avg_pool2d(out, 16), F.avg_pool2d(lq, 16))
        if args.w_barrier > 0:
            loss = loss + args.w_barrier * (F.relu(g_lp - args.lpips_cap) ** 2 + F.relu(g_ds - args.dists_cap) ** 2)
        iq = out[:min(args.iqa_batch, out.shape[0])]
        if args.iqa_eot > 0:
            def _eot(x):
                B, C, H, W = x.shape; outs = []
                for _ in range(args.iqa_eot):
                    dx, dy = int(torch.randint(-12, 13, (1,))), int(torch.randint(-12, 13, (1,)))
                    y = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
                    if torch.rand(1).item() < 0.5: y = torch.flip(y, dims=[3])
                    sc = float(torch.empty(1).uniform_(0.85, 1.0))
                    if sc < 0.98:
                        h2, w2 = int(H * sc) // 8 * 8, int(W * sc) // 8 * 8
                        y = F.interpolate(y, size=(h2, w2), mode="bilinear", align_corners=False, antialias=True)
                    outs.append(y)
                return outs
            iq_views = _eot(iq)
        else:
            iq_views = [iq]
        r = torch.tensor(0.0, device=DEV)
        if args.w_iqa > 0 and step >= args.iqa_start:
            which = step % 3
            def _sc(name, x):
                v = torch.stack([M[name](vw).mean() for vw in iq_views]).mean()
                return torch.clamp(v, max=caps[name]) if caps is not None else v
            if which == 0: r = 3.0 * args.w_clipiqa * _sc("clipiqa", iq)
            elif which == 1: r = 3.0 * args.w_maniqa * _sc("maniqa", iq)
            else: r = 3.0 * args.w_musiq * (_sc("musiq", iq) / 100.0)
            loss = loss - args.w_iqa * r
            if args.w_niqe > 0: loss = loss + args.w_niqe * M["niqe"](iq.clamp(0, 1)).mean() / 10.0
        gan_on = args.w_gan > 0 and step >= args.gan_start
        if gan_on: loss = loss + args.w_gan * F.softplus(-disc(out, lq)).mean()
        if args.w_hf_over > 0:
            hf = gt_aware_hf_overshoot_loss(out[:iq.shape[0]], gt[:iq.shape[0]])
            hf = hf[0] if isinstance(hf, tuple) else hf
            if torch.is_tensor(hf): loss = loss + args.w_hf_over * hf
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
        with torch.no_grad():
            for e, p in zip(ema, params): e.mul_(args.ema_decay).add_(p.detach(), alpha=1 - args.ema_decay)
        if gan_on:
            d_loss = F.softplus(disc(out.detach(), lq)).mean() + F.softplus(-disc(te, lq)).mean()
            d_opt.zero_grad(set_to_none=True); d_loss.backward(); d_opt.step()
        if step % 50 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} feat={f_loss.item():.4f} 教师L1={t_l1.item():.4f} 教师LPIPS={t_lp.item():.4f} "
                  f"GT_LPIPS={g_lp.item():.4f} GT_DISTS={g_ds.item():.4f} iqa={r.item():.3f} lr={lr_at(step):.1e} "
                  f"{(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            swap_ema(True)
            ev = evaluate_model(model, val_items, M, device=DEV, dtype=DT, robust_beta=0.15); ev.pop("rows", None)
            hist.append({"step": step, "score": float(ev["score"]), **{k: float(v) for k, v in ev["metrics"].items()}})
            flag = ""
            if ev["robust_score"] > best:
                best = ev["robust_score"]; flag = " <- best"
                torch.save({"cfg": model.cfg, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "step": step, "args": vars(args)}, os.path.join(args.out, "best.pth"))
            torch.save({"cfg": model.cfg, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "step": step, "args": vars(args)}, os.path.join(args.out, "last.pth"))
            swap_ema(False)
            print(f"[val {step}] score={ev['score']:.4f} robust={ev['robust_score']:.4f} PSNR={ev['metrics']['psnr']:.3f}{flag}", flush=True)
            json.dump(hist, open(os.path.join(args.out, "log.json"), "w"), indent=2)
    print(f"done. best robust={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
