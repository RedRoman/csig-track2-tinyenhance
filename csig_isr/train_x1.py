"""x1 画质增强训练：Samsung SRNet + 残差结构。

架构：out = clamp(LQ + alpha * (decoder(refine(net(LQ))) - 0.5))
  - 低频经 identity 无损通过
  - alpha 初始 0.2，训练起步时输出接近 LQ，不会一开始就崩
  - net 必须重训（原权重是 x4 SR 的，与 x1 语义不匹配，实测差 11.3 dB）

数据：DIV2K-Val/HR + DRealSR/HR + RealSR/HR 共 3193 张 512x512
退化：标定所得链条 双边(sigma 40-100) -> 高斯(sigma 0.8-1.6) -> JPEG(q 50-95)，参数随机
"""
import os, sys, io, time, glob, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

sys.path.insert(0, "/cache/zihaohe/csig_isr/NTIRE2026")
from diffusers import UNet2DConditionModel
from models.team11_SamsungAKCamera.model import create_srnet

ROOT = "/cache/zihaohe/csig_isr"
CKPT = f"{ROOT}/NTIRE2026/model_zoo/team11_SamsungAICamera/team11_mobile_samsung/weights.pth"
X1 = f"{ROOT}/x1_model/srnet_x1_init.pth"
SD21 = f"{ROOT}/sd21base"
VAL = f"{ROOT}/data/赛题二/验证集"


def degrade(img_u8, rng):
    """标定所得退化链：双边 -> 高斯低通 -> JPEG。img_u8: HxWx3 uint8"""
    x = img_u8
    if rng.random() < 0.9:
        s = float(rng.uniform(40, 100))
        x = cv2.bilateralFilter(x, 9, s, s)
    sigma = float(rng.uniform(0.8, 1.6))
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)
    x = np.clip(x, 0, 255).astype(np.uint8)
    q = int(rng.uniform(50, 95))
    buf = io.BytesIO()
    Image.fromarray(x).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


class PairDataset(Dataset):
    def __init__(self, paths, size=512, repeat=1):
        self.paths = paths * repeat
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.RandomState((int(torch.initial_seed()) % 100003 * 7919 + i * 104729) % (2 ** 31))
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
        lq = degrade(gt, rng)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        return to_t(lq), to_t(gt)


class ResSRNet(nn.Module):
    """残差包装：out = LQ + zero_conv(decoder(...) - 0.5)

    zero_conv 权重与偏置初始化为 0（ControlNet 思路），使训练起点的残差恒为 0，
    即 out == LQ，PSNR 精确等于 LQ 基线，绝不会先破坏再修。
    梯度仍可正常回传（zero_conv 自身的权重梯度非零）。
    """

    def __init__(self, net, refine, decoder, alpha=0.0):
        super().__init__()
        self.net, self.refine, self.decoder = net, refine, decoder
        self.zero_conv = nn.Conv2d(3, 3, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x):
        z = self.net(x)
        z = self.refine(z)
        d = self.decoder(z)
        return (x + self.zero_conv(d - 0.5)).clamp(0, 1)

    @property
    def alpha(self):
        """兼容日志：报告 zero_conv 权重范数作为残差强度指标。"""
        return self.zero_conv.weight.detach().abs().mean()


def build(alpha=0.2):
    cfg = UNet2DConditionModel.load_config(SD21, subfolder="unet")
    m = create_srnet(UNet2DConditionModel.from_config(cfg), checkpoint_path=CKPT)
    m.net.body[1].conv_in = nn.Conv2d(192, 160, 3, padding=1)
    m.net.body[0] = nn.PixelUnshuffle(8)
    sd = torch.load(X1, map_location="cpu", weights_only=False)
    m.net.load_state_dict(sd["net"])
    m.refine.load_state_dict(sd["refine"])
    m.decoder.load_state_dict(sd["decoder"])
    return ResSRNet(m.net, m.refine, m.decoder, alpha)


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def val_real(model, dev, n_crop=6):
    """在 3 组真实配对上评 PSNR（512 crop）。"""
    Image.MAX_IMAGE_PIXELS = None
    pairs = [("case1_lq.jpg", "case1_gt.png", 101), ("case2_lq.jpg", "case2_gt.png", 202),
             ("case3_lq.jpg", "case3_gt.jpg", 303)]
    p_in, p_out = [], []
    model.eval()
    for lqn, gtn, seed in pairs:
        lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
        if lq_f.shape != gt_f.shape:
            continue
        rng = np.random.RandomState(seed)
        _, _, H, W = gt_f.shape
        got, tries = 0, 0
        while got < n_crop and tries < n_crop * 12:
            tries += 1
            y, x = rng.randint(0, H - 512), rng.randint(0, W - 512)
            gt = gt_f[:, :, y:y + 512, x:x + 512].to(dev)
            if gt.std().item() < 0.045:
                continue
            got += 1
            lq = lq_f[:, :, y:y + 512, x:x + 512].to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(lq)
            out = out.float()
            p_in.append(-10 * np.log10(F.mse_loss(lq, gt).item()))
            p_out.append(-10 * np.log10(F.mse_loss(out, gt).item() + 1e-12))
    model.train()
    return float(np.mean(p_in)), float(np.mean(p_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--w_l1", type=float, default=1.0)
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_dists", type=float, default=2.0)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default=f"{ROOT}/train_out/run1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")

    paths = (sorted(glob.glob(f"{ROOT}/AdcSR/testset/DIV2K-Val/HR/*.png"))
             + sorted(glob.glob(f"{ROOT}/AdcSR/testset/DRealSR/HR/*.png"))
             + sorted(glob.glob(f"{ROOT}/AdcSR/testset/RealSR/HR/*.png")))
    print(f"训练图 {len(paths)} 张", flush=True)

    ds = PairDataset(paths, 512, repeat=20)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)

    model = build(args.alpha).to(dev)
    model.train()
    print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)

    import pyiqa
    lpips_fn = pyiqa.create_metric("lpips", device=dev, as_loss=True)
    dists_fn = pyiqa.create_metric("dists", device=dev, as_loss=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda")

    pi0, po0 = val_real(model, dev)
    print(f"[step 0] 真实验证集 LQ={pi0:.3f} dB  模型={po0:.3f} dB  增益 {po0 - pi0:+.3f}", flush=True)

    best, step, t0, log = -1e9, 0, time.time(), []
    while step < args.steps:
        for lq, gt in dl:
            if step >= args.steps:
                break
            lq, gt = lq.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(lq)
            out = out.float()
            l1 = F.l1_loss(out, gt)
            lp = lpips_fn(out, gt).mean()
            dd = dists_fn(out, gt).mean()
            loss = args.w_l1 * l1 + args.w_lpips * lp + args.w_dists * dd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            if step % 50 == 0:
                print(f"[{step:>5}/{args.steps}] loss {loss.item():.4f} "
                      f"(l1 {l1.item():.4f} lpips {lp.item():.4f} dists {dd.item():.4f}) "
                      f"|zc| {model.alpha.item():.5f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / step:.2f}s/it", flush=True)

            if step % args.val_every == 0:
                pi, po = val_real(model, dev)
                log.append({"step": step, "lq": pi, "model": po, "loss": loss.item()})
                flag = ""
                if po > best:
                    best = po
                    torch.save({"net": model.net.state_dict(),
                                "refine": model.refine.state_dict(),
                                "decoder": model.decoder.state_dict(),
                                "zero_conv": model.zero_conv.state_dict(),
                                "step": step},
                               f"{args.out}/best.pth")
                    flag = "  <- best, saved"
                print(f"[val {step}] LQ={pi:.3f}  模型={po:.3f}  增益 {po - pi:+.3f} dB{flag}",
                      flush=True)
                json.dump(log, open(f"{args.out}/log.json", "w"), indent=1)

    torch.save({"net": model.net.state_dict(), "refine": model.refine.state_dict(),
                "decoder": model.decoder.state_dict(),
                "zero_conv": model.zero_conv.state_dict(),
                "step": step}, f"{args.out}/final.pth")
    print(f"训练完成，best 真实验证 PSNR = {best:.3f} dB", flush=True)


if __name__ == "__main__":
    main()
