"""TinySR x1 微调训练。

架构（x1，零改造）：
    LQ 512 -> VAE.encode -> 64x64x16 latent
    -> DiT 预测残差 -> z_out = z_in - z_pred
    -> VAE.decode -> 512

训练策略：
  1. 先 warmup 触发 TinySR 的惰性缓存（首次 forward 会算出 AdaLN 的 scale/shift
     并 del time_text_embed / norm_out / 各 block 的 norm1）。因 timestep 与
     prompt embedding 固定，该缓存数学上等价，且让 AdaLN 调制参数固定。
  2. 只训 LoRA（attention/FFN 上的 rank-64 适配器，30.6M），其余冻结。
     zero-shot 已达 Score 3.33，只需适配"过度降噪"退化，不需要动主干。

数据与退化链与 Samsung 实验完全一致，保证可比。
"""
import os, sys, io, time, glob, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

ROOT = "/cache/zihaohe/csig_isr"
TS = f"{ROOT}/TinySR"
sys.path.insert(0, TS)
os.chdir(TS)

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
from models.discriminator import ProjectedDiscriminator


def _as_list(o):
    """ProjectedDiscriminator 返回嵌套 tuple，展平成张量列表。"""
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, (list, tuple)):
        out = []
        for t in o:
            out += _as_list(t)
        return out
    return []

VAL = f"{ROOT}/data/赛题二/验证集"
dev = torch.device("cuda")


def degrade(img_u8, rng):
    """标定所得退化链：双边 -> 高斯低通 -> JPEG。"""
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
        self.paths, self.size = paths * repeat, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.RandomState((int(torch.initial_seed()) % 100003 * 7919 + i * 104729) % (2 ** 31))
        gt = np.asarray(Image.open(self.paths[i]).convert("RGB"), dtype=np.uint8)
        h, w, _ = gt.shape
        s = self.size
        if h > s or w > s:
            y, x = rng.randint(0, max(h - s, 0) + 1), rng.randint(0, max(w - s, 0) + 1)
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


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


class DenseBlock(nn.Module):
    """FiDeSR 的稠密块（照 repos/FiDeSR/fidesr.py 源码）。"""

    def __init__(self, in_ch, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_ch + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_ch + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_ch + 4 * growth, in_ch, 3, 1, 1)
        self.act = nn.SiLU()
        nn.init.zeros_(self.conv5.weight)
        nn.init.zeros_(self.conv5.bias)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        return self.conv5(torch.cat([x, x1, x2, x3, x4], 1))


class LRRB(nn.Module):
    """Latent Residual Refinement Block（移植自 FiDeSR, CVPR2026）。

    输入 concat([z, pred])，输出对 pred 的修正量 delta_r。
    out_conv 零初始化 => 起点 delta_r == 0，模型精确等价于未加 LRRB 的版本。
    """

    def __init__(self, in_ch=32, mid_ch=64, growth=32, res_scale=0.2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, mid_ch, 1, 1, 0)
        self.db1 = DenseBlock(mid_ch, growth)
        self.db2 = DenseBlock(mid_ch, growth)
        self.db3 = DenseBlock(mid_ch, growth)
        self.out_conv = nn.Conv2d(mid_ch, in_ch // 2, 1, 1, 0)
        self.res_scale = res_scale
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_cat):
        # LRRB 参数为 fp32（训练精度），输入 latent 为 fp16 —— 在此对齐并转回
        dt_in = x_cat.dtype
        f = self.in_conv(x_cat.to(self.in_conv.weight.dtype))
        f = f + self.db1(f) * self.res_scale
        f = f + self.db2(f) * self.res_scale
        f = f + self.db3(f) * self.res_scale
        return self.out_conv(f).to(dt_in)


class TinySRx1(nn.Module):
    """把 TinySR 的三段封成一个 module，便于统一 forward。"""

    def __init__(self, vae, tr, timesteps, pooled, lrrb=None):
        super().__init__()
        self.vae, self.tr, self.lrrb = vae, tr, lrrb
        self.register_buffer("ts", timesteps, persistent=False)
        self.register_buffer("pooled", pooled, persistent=False)

    def forward(self, lq01):
        x = lq01 * 2 - 1
        sf = self.vae.config.scaling_factor
        z = self.vae.encode(x).latents * sf
        pred = self.tr(hidden_states=z, timestep=self.ts,
                       pooled_projections=self.pooled, return_dict=False)[0]
        if self.lrrb is not None:
            pred = pred + self.lrrb(torch.cat([z, pred], dim=1))
        img = self.vae.decode((z - pred) / sf, return_dict=False)[0]
        return (img * 0.5 + 0.5).clamp(0, 1)


def build(train_vae_decoder=False, use_lrrb=False, init_ckpt=None):
    dt = torch.float16
    tr = TinySD3Transformer2DModel.from_pretrained(
        f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr", subfolder="transformer",
        torch_dtype=dt, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    vae = AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable", torch_dtype=dt)
    cfg = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian",
                     target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj",
                                     "linear", "linear_1", "linear_2", "net.2"])
    tr.add_adapter(cfg)
    tr.enable_adapters()
    load_lora_state_dict(StableDiffusion3Pipeline.lora_state_dict(
        f"{TS}/checkpoint/tinysr", weight_name="transformer.safetensors"), tr)
    vae = vae.to(dev, dtype=dt)
    tr = tr.to(dev, dtype=dt)

    ts = torch.tensor([1000.0], device=dev, dtype=dt)
    pooled = torch.load(f"{TS}/dataset/default/pool_embeds.pt", map_location=dev).to(dtype=dt)

    # warmup：触发惰性缓存（必须 no_grad，否则 scale/shift 会带上计算图）
    tr.eval()
    with torch.no_grad():
        w = torch.rand(1, 3, 512, 512, device=dev, dtype=dt) * 2 - 1
        zz = vae.encode(w).latents * vae.config.scaling_factor
        _ = tr(hidden_states=zz, timestep=ts, pooled_projections=pooled, return_dict=False)
    assert getattr(tr, "initialized", False), "惰性缓存未触发"
    print("  惰性缓存已触发（AdaLN scale/shift 固定）", flush=True)

    if init_ckpt and os.path.exists(init_ckpt):
        _sd = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        _cur = tr.state_dict(); _hit = 0
        for _k, _v in _sd.get("lora", {}).items():
            if _k in _cur and _cur[_k].shape == _v.shape:
                _cur[_k].copy_(_v.to(_cur[_k].dtype)); _hit += 1
        print(f"  续训自 {init_ckpt} (step {_sd.get('step')}) LoRA {_hit} 项", flush=True)

    lrrb = None
    if use_lrrb:
        # latent 16 通道 -> concat 后 32 通道
        lrrb = LRRB(in_ch=2 * vae.config.latent_channels, mid_ch=64,
                    growth=32, res_scale=0.2).to(dev)  # 保持 fp32
    model = TinySRx1(vae, tr, ts, pooled, lrrb)

    # 冻结全部，只放开 LoRA（+ 可选 VAE decoder）
    for p in model.parameters():
        p.requires_grad_(False)
    n_lora = 0
    for n, p in model.tr.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            p.data = p.data.float()          # LoRA 用 fp32 训，主干保持 fp16
            n_lora += p.numel()
    n_lrrb = 0
    if model.lrrb is not None:
        for p in model.lrrb.parameters():
            p.requires_grad_(True)
            p.data = p.data.float()
            n_lrrb += p.numel()
    n_vae = 0
    if train_vae_decoder:
        for n, p in model.vae.named_parameters():
            if n.startswith("decoder"):
                p.requires_grad_(True)
                p.data = p.data.float()
                n_vae += p.numel()
    print(f"  可训练：LoRA {n_lora/1e6:.2f} M"
          + (f" + LRRB {n_lrrb/1e6:.2f} M" if n_lrrb else "")
          + (f" + VAE decoder {n_vae/1e6:.2f} M" if n_vae else ""), flush=True)
    return model


_VAL_METRICS = {}


def _get_val_metrics(dev):
    """惰性创建 Score 所需的 6 个指标（与 NTIRE eval.py 同口径）。"""
    global _VAL_METRICS
    if not _VAL_METRICS:
        import pyiqa
        _VAL_METRICS = {
            "lpips": pyiqa.create_metric("lpips", device=dev),
            "dists": pyiqa.create_metric("dists", device=dev),
            "niqe": pyiqa.create_metric("niqe", device=dev),
            "musiq": pyiqa.create_metric("musiq", device=dev),
            "maniqa": pyiqa.create_metric("maniqa-pipal", device=dev),
            "clipiqa": pyiqa.create_metric("clipiqa", device=dev),
        }
    return _VAL_METRICS


_VAL_CACHE = None


def _build_val_set(n_per=16):
    """按退化程度分层的验证集（轻/中/重各取），避免只看简单样本产生误判。

    教训：先前只用 18 个随机 crop，且集中在纹理简单区域，导致
    "视觉验证通过"的错误结论 —— 重度退化的 case3 其实全档位都在编造伪纹理。
    """
    global _VAL_CACHE
    if _VAL_CACHE is not None:
        return _VAL_CACHE
    Image.MAX_IMAGE_PIXELS = None
    pairs = [("case1_lq.jpg", "case1_gt.png", 101), ("case2_lq.jpg", "case2_gt.png", 202),
             ("case3_lq.jpg", "case3_gt.jpg", 303)]
    items = []
    for lqn, gtn, seed in pairs:
        lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
        if lq_f.shape != gt_f.shape:
            continue
        rng = np.random.RandomState(seed)
        _, _, H, W = gt_f.shape
        got, tries = 0, 0
        while got < n_per and tries < n_per * 15:
            tries += 1
            y, x = rng.randint(0, H - 512), rng.randint(0, W - 512)
            gt = gt_f[:, :, y:y + 512, x:x + 512]
            if gt.std().item() < 0.035:
                continue
            got += 1
            lq = lq_f[:, :, y:y + 512, x:x + 512]
            psnr_lq = -10 * np.log10(F.mse_loss(lq, gt).item() + 1e-12)
            items.append({"lq": lq.clone(), "gt": gt.clone(), "psnr_lq": psnr_lq})
    ps = sorted(i["psnr_lq"] for i in items)
    lo, hi = ps[len(ps) // 3], ps[2 * len(ps) // 3]
    for it in items:
        it["tier"] = "重" if it["psnr_lq"] < lo else ("轻" if it["psnr_lq"] > hi else "中")
    _VAL_CACHE = items
    from collections import Counter
    c = Counter(i["tier"] for i in items)
    print(f"  验证集 {len(items)} crop  分层: 重{c['重']} 中{c['中']} 轻{c['轻']}  "
          f"(LQ PSNR 分界 {lo:.2f}/{hi:.2f})", flush=True)
    return _VAL_CACHE


@torch.no_grad()
def val_real(model, n_crop=6):
    items = _build_val_set()
    p_in, p_out, s_out = [], [], []
    tier_score = {"重": [], "中": [], "轻": []}
    was = model.training
    model.eval()
    for it in items:
        gt = it["gt"].to(dev)
        lq = it["lq"].to(dev)
        out = model(lq.half()).float()
        p_in.append(it["psnr_lq"])
        p_out.append(-10 * np.log10(F.mse_loss(out, gt).item() + 1e-12))
        _M = _get_val_metrics(dev)
        _s = ((1 - _M["lpips"](out, gt).item()) + (1 - _M["dists"](out, gt).item())
              + _M["clipiqa"](out).item() + _M["maniqa"](out).item()
              + _M["musiq"](out).item() / 100
              + max(0, (10 - _M["niqe"](out).item()) / 10))
        s_out.append(_s)
        tier_score[it["tier"]].append(_s)
        del gt, lq, out
        torch.cuda.empty_cache()
    if was:
        model.train()
    tiers = {k: (float(np.mean(v)) if v else 0.0) for k, v in tier_score.items()}
    return float(np.mean(p_in)), float(np.mean(p_out)), float(np.mean(s_out)), tiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--w_l1", type=float, default=1.0)
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_dists", type=float, default=2.0)
    ap.add_argument("--w_fid", type=float, default=2.0,
                    help="低频保真约束权重，抑制信息不足区域的幻觉")
    ap.add_argument("--w_iqa", type=float, default=0.5,
                    help="无参考 IQA(CLIPIQA+MUSIQ/100) 的负向权重，0 表示关闭")
    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train_vae_decoder", action="store_true")
    ap.add_argument("--use_lrrb", action="store_true",
                    help="插入 LRRB latent 精修模块（移植自 FiDeSR）")
    ap.add_argument("--w_gan", type=float, default=0.0,
                    help="对抗损失权重，0 表示不用 GAN")
    ap.add_argument("--d_lr", type=float, default=1e-4, help="判别器学习率")
    ap.add_argument("--gan_start", type=int, default=200,
                    help="前 N 步不开 GAN，让生成器先稳定")
    ap.add_argument("--extra_data", type=str, default=None,
                    help="额外训练集目录，逗号分隔（如 LSDIR 切好的 512 patch）")
    ap.add_argument("--init_ckpt", type=str, default=None,
                    help="从已有 checkpoint 续训")
    ap.add_argument("--out", type=str, default=f"{ROOT}/train_out/tinysr1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = (sorted(glob.glob(f"{ROOT}/AdcSR/testset/DIV2K-Val/HR/*.png"))
             + sorted(glob.glob(f"{ROOT}/AdcSR/testset/DRealSR/HR/*.png"))
             + sorted(glob.glob(f"{ROOT}/AdcSR/testset/RealSR/HR/*.png")))
    n_base = len(paths)
    if args.extra_data:
        for d in args.extra_data.split(","):
            d = d.strip()
            if d:
                ex = sorted(glob.glob(os.path.join(d, "*.png")))
                paths += ex
                print(f"  + {os.path.basename(d)}: {len(ex)} 张", flush=True)
    print(f"训练图 {len(paths)} 张（基础 {n_base}）", flush=True)

    # 数据量大时不需要高 repeat（原 3193 张要 repeat=20 才够一轮训练）
    rep = 20 if len(paths) < 8000 else max(1, 60000 // max(len(paths), 1))
    print(f"  repeat={rep}, 等效样本 {len(paths)*rep}", flush=True)
    dl = DataLoader(PairDataset(paths, 512, repeat=rep), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True,
                    persistent_workers=True)

    model = build(args.train_vae_decoder, args.use_lrrb, args.init_ckpt)

    import pyiqa
    lpips_fn = pyiqa.create_metric("lpips", device=dev, as_loss=True)
    dists_fn = pyiqa.create_metric("dists", device=dev, as_loss=True)
    # 无参考 IQA 直接进损失（Score 里 CLIPIQA/MANIQA/MUSIQ 各占 1 分权重）
    # SamsungAICamera 在同构赛题用同样做法，NTIRE 组织方明确裁定合规
    clipiqa_fn = pyiqa.create_metric("clipiqa", device=dev, as_loss=True) if args.w_iqa > 0 else None
    musiq_fn = pyiqa.create_metric("musiq", device=dev, as_loss=True) if args.w_iqa > 0 else None

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.05)

    # 判别器（DINO 特征 + 多层 head，DINO 冻结只训 head 7.39M）
    D = D_opt = None
    if args.w_gan > 0:
        D = ProjectedDiscriminator(c_dim=0).to(dev)
        D.train()
        d_params = [p for p in D.parameters() if p.requires_grad]
        D_opt = torch.optim.AdamW(d_params, lr=args.d_lr, betas=(0.0, 0.99))
        print(f"  判别器可训练 {sum(p.numel() for p in d_params)/1e6:.2f} M，"
              f"GAN 从 step {args.gan_start} 开始", flush=True)

    pi0, po0, so0, tiers0 = val_real(model)
    print(f"[step 0] LQ={pi0:.3f}dB 模型={po0:.3f}dB Score={so0:.4f} "
              f"(重{tiers0['重']:.3f} 中{tiers0['中']:.3f} 轻{tiers0['轻']:.3f})", flush=True)

    best, step, t0, log = -1e9, 0, time.time(), []
    model.train()
    while step < args.steps:
        for lq, gt in dl:
            if step >= args.steps:
                break
            lq, gt = lq.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
            out = model(lq.half()).float()
            l1 = F.l1_loss(out, gt)
            lp = lpips_fn(out, gt).mean()
            dd = dists_fn(out, gt).mean()
            loss = args.w_l1 * l1 + args.w_lpips * lp + args.w_dists * dd

            # ---- 对抗损失（生成器侧）----
            gan_val = 0.0
            if D is not None and step >= args.gan_start:
                c0 = torch.zeros(out.shape[0], 0, device=dev)
                fake_logits = _as_list(D(out * 2 - 1, c0))
                g_adv = sum(-t.mean() for t in fake_logits) / max(len(fake_logits), 1)
                loss = loss + args.w_gan * g_adv
                gan_val = g_adv.item()
            # 保真约束：输出的低频必须贴近 LQ（LQ 低频实测无损，能量比恒为 1.00）
            # 目的是让模型在信息不足时退化为温和增强，而非编造伪纹理
            fid_val = 0.0
            if args.w_fid > 0:
                k = 16
                lo_out = F.avg_pool2d(out, k)
                lo_lq = F.avg_pool2d(lq, k)
                fid = F.l1_loss(lo_out, lo_lq)
                loss = loss + args.w_fid * fid
                fid_val = fid.item()
            iqa_val = 0.0
            if clipiqa_fn is not None:
                # 这两项越大越好，取负号作为损失
                ci = clipiqa_fn(out).mean()
                mu = musiq_fn(out).mean() / 100.0
                loss = loss - args.w_iqa * (ci + mu)
                iqa_val = (ci + mu).item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            # ---- 判别器更新（hinge loss）----
            d_val = 0.0
            if D is not None and step >= args.gan_start:
                c0 = torch.zeros(gt.shape[0], 0, device=dev)
                real_logits = _as_list(D(gt * 2 - 1, c0))
                fake_logits = _as_list(D(out.detach() * 2 - 1, c0))
                d_real = sum(F.relu(1 - t).mean() for t in real_logits) / max(len(real_logits), 1)
                d_fake = sum(F.relu(1 + t).mean() for t in fake_logits) / max(len(fake_logits), 1)
                d_loss = d_real + d_fake
                D_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(d_params, 1.0)
                D_opt.step()
                d_val = d_loss.item()

            step += 1

            if step % 50 == 0:
                print(f"[{step:>5}/{args.steps}] loss {loss.item():.4f} "
                      f"(l1 {l1.item():.4f} lpips {lp.item():.4f} dists {dd.item():.4f} "
                      f"iqa {iqa_val:.4f} fid {fid_val:.4f} gan {gan_val:+.3f} D {d_val:.3f}) "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/step:.2f}s/it", flush=True)

            if step % args.val_every == 0:
                pi, po, so, tiers = val_real(model)
                log.append({"step": step, "lq": pi, "psnr": po, "score": so, "tiers": tiers, "loss": loss.item()})
                flag = ""
                if so > best:
                    best = so
                    sd = {k: v for k, v in model.tr.state_dict().items() if "lora_" in k}
                    torch.save({"lora": sd,
                                "lrrb": (model.lrrb.state_dict() if model.lrrb is not None else None),
                                "vae_decoder": ({k: v for k, v in model.vae.state_dict().items()
                                                 if k.startswith("decoder")}
                                                if args.train_vae_decoder else None),
                                "step": step}, f"{args.out}/best.pth")
                    if D is not None:
                        torch.save({"D": D.heads.state_dict(), "step": step},
                                   f"{args.out}/best_D.pth")
                    flag = "  <- best, saved"
                print(f"[val {step}] PSNR={po:.3f}dB Score={so:.4f} "
                      f"(重{tiers['重']:.3f} 中{tiers['中']:.3f} 轻{tiers['轻']:.3f}){flag}", flush=True)
                json.dump(log, open(f"{args.out}/log.json", "w"), indent=1)

    sd = {k: v for k, v in model.tr.state_dict().items() if "lora_" in k}
    torch.save({"lora": sd,
                "lrrb": (model.lrrb.state_dict() if model.lrrb is not None else None),
                "step": step}, f"{args.out}/final.pth")
    print(f"训练完成，best Score = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
