"""在 CSIG 赛题二上微调 VOSR 0.5B。

此前所有 VOSR 结论都只来自零样本推理 + 时延剖析，从未训过一步。这个脚本补上。

首要目标不是拿最高分，是测出 **帕累托交换率**：
  TinySR 的每次训练都是严格此消彼长（+0.19 MANIQA 换 −0.04 LPIPS），
  这是容量受限的特征。若 VOSR 的交换率明显更平，说明它的前沿更靠外，
  35ms 下 +0.291 的回本线就有希望；若一样陡，VOSR 结案。

复用我们已验证的组件：PairDataset（退化链）、create_metrics（损失与评测）、
evaluate_model（验证）。只把模型换成 VOSR。
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

VOSR_DIR = "/cache/zihaohe/VOSR"
CODE_DIR = "/cache/zihaohe/csig_isr/v3/code"
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, VOSR_DIR)
# 注意：不要动 TORCH_HOME。pyiqa 的 lpips/dists/musiq/maniqa 权重都在默认的
# ~/.cache/torch/hub/checkpoints 下，改了会触发全部重新下载（踩过一次，浪费 20 分钟）。
# DINOv2 我们是按绝对路径手动加载的，不依赖 TORCH_HOME。
# LightningDiT 的 block 上带 @torch.compile，inductor 在这套环境里报
# "'float' object has no attribute 'meta'"，直接关掉走 eager。
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import cv2
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from safetensors.torch import load_file

from settings import ROOT
from metrics_utils import create_metrics
from eval_utils import collect_crops, evaluate_model
from degradation import degrade as degrade_legacy

DEV, DT = torch.device("cuda"), torch.float16
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def mosaic_in(t, mode):
    """把 512 输出平铺成 4K 尺寸，让 MUSIQ 落回官方评测的响应区间。可导。
    尺寸扫描实测：3072x2304 时与真实 4K 排序一致（r=+0.928），
    2048x1536 还在错误一侧（r=-0.634）。4096x3072 显存扛不住。"""
    if mode != "mosaic4k":
        return t
    t = t[:1]
    H = int(os.environ.get("CSIG_MOSAIC_H", 2304))
    W = int(os.environ.get("CSIG_MOSAIC_W", 3072))
    ry = -(-H // t.shape[-2]); rx = -(-W // t.shape[-1])
    return t.repeat(1, 1, ry, rx)[:, :, :H, :W]


class PairDataset(Dataset):
    """与 train_tinysr.py 里的同名类逐字一致（不能 import：两边都有 models 包，会撞名）"""

    def __init__(self, paths, size=512, repeat=1, mix=0.0):
        self.paths = paths * repeat
        self.size = int(size)
        self.mix = float(mix)

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
        # 退化对齐（degrade_probe 标定）：现行链 DISTS 0.228 vs 真实域 0.294，
        # 按 mix 概率混入加强档（σ1.8/q55/bil110），让势垒在真实域等效难度上咬合。
        if self.mix > 0 and rng.random() < self.mix:
            img = cv2.bilateralFilter(gt, d=9, sigmaColor=110, sigmaSpace=110)
            img = cv2.GaussianBlur(img, (9, 9), 1.8)
            enc = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 55])[1]
            lq = np.ascontiguousarray(cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1])
        else:
            lq = degrade_legacy(gt, rng)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        return to_t(lq), to_t(gt)


class Args:
    """VOSR 的 args.json 转成属性访问，并补上 tiled_vae 需要的字段"""
    def __init__(self, d):
        self.__dict__.update(d)
        self.vae_tile_size = 0
        self.tile_size = 0
        self.posterior_mode = True


class VOSRWrapper(nn.Module):
    """把 VOSR 的四段流程包成 lq01 -> sr01，接口与 TinySR 的 model 一致。

    可导路径只经过 DiT 和 LightDecoder；VAE 编码与 DINOv2 都在输入侧，冻结。
    """

    def __init__(self, dit, vae, light_decoder, venc, a, fixed_noise=True):
        super().__init__()
        self.tr = dit                 # 沿用 TinySR 那边的命名，便于复用工具
        self.vae = vae
        self.dec = light_decoder      # qwen 分支为 None，直接用 vae.decode
        self.venc = venc
        self.a = a
        self.qwen = (getattr(a, "ae_type", "sd2") == "qwen")
        if self.qwen:
            # 16 通道 latent 走逐通道标准化：(z-mean)*std_inv，解码时反过来
            self.register_buffer("lat_mean",
                                 torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1))
            self.register_buffer("lat_std_inv",
                                 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1))
        self.alpha_gate = None
        self.register_buffer("fixed_alpha", torch.tensor(1.0))
        self.fixed_noise = fixed_noise
        self._noise = None

    @torch.no_grad()
    def _cond(self, lq11):
        raw = (0.5 * lq11 + 0.5) * 255.
        x = raw / 255.
        x = F.interpolate(x.float(), self.a.dinov2_size, mode="bicubic").clip(0., 1.)
        x = Normalize(IMNET_MEAN, IMNET_STD)(x).to(lq11.dtype)
        feats, x_norm = self.venc.forward_with_features(x)
        z = [v for k, v in feats.items() if k.startswith("layer_")]
        z[-1] = x_norm
        return [z[i] for i in self.a.layer_dinov2b_list]

    @torch.no_grad()
    def _encode(self, lq11):
        z = self.vae.encode(lq11).latent_dist.mode()
        if self.qwen:
            return (z - self.lat_mean.to(z.dtype)) * self.lat_std_inv.to(z.dtype)
        return z * self.vae.config.scaling_factor

    def _decode(self, z):
        if self.qwen:
            zz = z / self.lat_std_inv.to(z.dtype) + self.lat_mean.to(z.dtype)
            return self.vae.decode(zz, return_dict=False)[0].clamp(-1, 1)
        return self.dec(z / self.vae.config.scaling_factor).clamp(-1, 1)

    def _impl(self, lq01):
        lq11 = lq01 * 2 - 1
        lat = self._encode(lq11)
        cond = self._cond(lq11)
        # 一步流匹配：t 从 1 到 0。噪声固定住，否则损失里全是采样噪声，
        # 交换率这种要比斜率的量会被淹掉。
        if self.fixed_noise:
            if self._noise is None or self._noise.shape != lat.shape:
                g = torch.Generator(device=lat.device).manual_seed(1234)
                self._noise = torch.randn(lat.shape, generator=g,
                                          device=lat.device, dtype=lat.dtype)
            z = self._noise
        else:
            z = torch.randn_like(lat)
        B = lat.shape[0]
        t1 = torch.ones(B, device=lat.device, dtype=lat.dtype)
        t0 = torch.zeros(B, device=lat.device, dtype=lat.dtype)
        u = self.tr(torch.cat([lat, z], 1), t1, t0, cond)
        sr = self._decode(z - u)
        return (sr * 0.5 + 0.5).clamp(0, 1), self.fixed_alpha.reshape(1, 1, 1, 1)

    def forward(self, lq01):
        return self._impl(lq01)[0]

    def forward_train(self, lq01):
        return self._impl(lq01)


def build(ckpt, train_dec: bool, dino_layers: int, dino_size: int):
    from models.lightningdit import LightningDiT

    a = Args(json.load(open(os.path.join(ckpt, "args.json"))))
    if dino_size:
        a.dinov2_size = int(dino_size)
    qwen = (a.ae_type == "qwen")
    base_ch = 16 if qwen else 4

    if qwen:
        # 1.4B 用 Qwen-Image 的 16 通道 VAE，且自带解码器（没有 LightDecoder）
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        vae = AutoencoderKLQwenImage2D.from_pretrained(
            os.path.join(VOSR_DIR, "preset/ckpts/Qwen-Image-vae-2d"),
            torch_dtype=DT).to(DEV).eval()
        dec = None
    else:
        from diffusers import AutoencoderKL
        from models.light_decoder import LightDecoder
        vae = AutoencoderKL.from_pretrained(
            os.path.join(VOSR_DIR, "preset/ckpts/stable-diffusion-2-1-base"),
            subfolder="vae", torch_dtype=DT).to(DEV).eval()
        ck = torch.load(os.path.join(VOSR_DIR, "preset/ckpts/sd21_lwdecoder.pth"),
                        map_location="cpu")
        c = ck["config"]
        dec = LightDecoder(in_channels=c["in_channels"], out_channels=c["out_channels"],
                           block_out_channels=tuple(c["block_out_channels"]),
                           layers_per_block=c["layers_per_block"])
        dec.load_state_dict(ck["model_state_dict"])
        dec = dec.to(DEV, dtype=DT)
        dec.train(train_dec)
    for p in vae.parameters(): p.requires_grad_(False)

    dino_tag = "dinov2_vitl14" if a.enc_type == "dinov2l" else "dinov2_vitb14"
    venc = torch.hub.load(os.path.join(VOSR_DIR, "preset/ckpts/torch_cache/facebookresearch_dinov2_main"),
                          dino_tag, source="local", pretrained=False)
    sd = torch.load(os.path.join(VOSR_DIR,
                    f"preset/ckpts/torch_cache/checkpoints/{dino_tag}_pretrain.pth"),
                    map_location="cpu")
    venc.load_state_dict(sd, strict=False)
    if hasattr(venc, "head"): venc.head = nn.Identity()
    need = max(a.layer_dinov2b_list) + 1
    keep = max(need, int(dino_layers)) if dino_layers else len(venc.blocks)
    if keep < len(venc.blocks):
        venc.blocks = nn.ModuleList(list(venc.blocks)[:keep])

    def fwf(self, x, masks=None):
        feats = {}
        x = self.prepare_tokens_with_masks(x, masks)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            feats[f"layer_{i}"] = x[:, 1:]
        return feats, self.norm(x)[:, 1:]
    venc.forward_with_features = fwf.__get__(venc)
    venc = venc.to(DEV, dtype=DT).eval()
    for p in venc.parameters(): p.requires_grad_(False)

    dit = LightningDiT(
        input_size=a.resolution // 8, patch_size=a.patch_size,
        in_channels=2 * base_ch, out_channels=base_ch,
        hidden_size=a.dim, depth=a.depth, num_heads=a.num_heads, mlp_ratio=a.mlp_ratio,
        z_dims=a.enc_dim, encdim_ratio=a.encdim_ratio, auxiliary_time_cond=False,
        use_qknorm=a.use_qknorm, use_swiglu=a.use_swiglu, use_rope=a.use_rope,
        use_rmsnorm=a.use_rmsnorm, wo_shift=a.wo_shift,
        num_fused_layers=len(a.layer_dinov2b_list))
    w = load_file(os.path.join(ckpt, "checkpoints", "ema_model.safetensors"))
    rep = dit.load_state_dict({k: v.to(torch.float32) for k, v in w.items()}, strict=False)
    print(f"DiT 载入：缺失 {len(rep.missing_keys)} 多余 {len(rep.unexpected_keys)}", flush=True)
    dit = dit.to(DEV, dtype=DT)
    if hasattr(dit, "forward_flexible"):
        dit.forward = dit.forward_flexible
    return VOSRWrapper(dit, vae, dec, venc, a).to(DEV)


def add_lora(dit, r, alpha):
    from peft import LoraConfig, get_peft_model
    names = set()
    for n, m in dit.named_modules():
        if isinstance(m, nn.Linear) and ("blocks." in n):
            names.add(n.split(".")[-1])
    targets = sorted(n for n in names if n not in ("adaLN_modulation",))
    print(f"LoRA 目标模块: {targets}", flush=True)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=targets)
    return get_peft_model(dit, cfg), targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/cache/zihaohe/VOSR_weights/VOSR_0.5B_os")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--train_decoder", action="store_true")
    ap.add_argument("--dec_lr", type=float, default=1e-5)
    ap.add_argument("--dino_layers", type=int, default=9)
    ap.add_argument("--dino_size", type=int, default=0, help="0=沿用 args.json 的 448")
    ap.add_argument("--w_l1", type=float, default=0.5)
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_dists", type=float, default=1.5)
    ap.add_argument("--w_fid", type=float, default=3.0)
    ap.add_argument("--w_iqa", type=float, default=3.0)
    ap.add_argument("--w_clipiqa", type=float, default=1.0)
    ap.add_argument("--w_maniqa", type=float, default=1.0)
    ap.add_argument("--w_musiq", type=float, default=1.0)
    # NIQE：VOSR 的 NR 里最弱的一项（4.778 vs TinySR run12 的 3.766），
    # 而 TinySR 那边它一直在损失里。补上。
    ap.add_argument("--w_niqe", type=float, default=0.0)
    # MUSIQ 的度量口径。实测 512 crop 与官方 4K 口径的排序相关是 -0.544（反向），
    # 平铺成 3072x2304 后是 +0.928。
    ap.add_argument("--musiq_proxy", choices=["crop", "mosaic4k"], default="crop")
    # FR 势垒：阈值以内不施力，越界后平方重罚，使 FR 成为边界而非可交易筹码。
    ap.add_argument("--lpips_cap", type=float, default=0.0)
    ap.add_argument("--dists_cap", type=float, default=0.0)
    ap.add_argument("--w_barrier", type=float, default=0.0)
    ap.add_argument("--iqa_batch", type=int, default=1)
    ap.add_argument("--val_every", type=int, default=100)
    ap.add_argument("--val_crops", type=int, default=12)
    ap.add_argument("--extra_data", default="/cache/zihaohe/csig_isr/train_data/lsdir512")
    ap.add_argument("--degrade_mix", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="/cache/zihaohe/csig_isr/train_out/vosr_ft")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = build(args.ckpt, args.train_decoder, args.dino_layers, args.dino_size)
    for p in model.parameters(): p.requires_grad_(False)
    model.tr, targets = add_lora(model.tr, args.lora_r, args.lora_alpha)
    model.tr = model.tr.to(DEV)
    if hasattr(model.tr, "base_model"):
        bm = model.tr.base_model.model
        if hasattr(bm, "forward_flexible"):
            bm.forward = bm.forward_flexible
    params = [p for p in model.tr.parameters() if p.requires_grad]
    for p in params: p.data = p.data.float()
    groups = [{"params": params, "lr": args.lr}]
    if args.train_decoder and model.dec is not None:
        for p in model.dec.parameters(): p.requires_grad_(True)
        dp = list(model.dec.parameters())
        for p in dp: p.data = p.data.float()
        groups.append({"params": dp, "lr": args.dec_lr})
    elif args.train_decoder:
        print("提示：qwen 分支没有独立 LightDecoder，--train_decoder 忽略", flush=True)
    n_tr = sum(p.numel() for g in groups for p in g["params"])
    print(f"可训练参数 {n_tr/1e6:.2f} M", flush=True)

    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    M = create_metrics(DEV, as_loss=True, include_psnr=True)

    # 与 train_tinysr.py 逐字一致：3193 张基础 + lsdir512 的 57300 张 = 60493
    paths = (sorted(glob.glob(os.path.join(ROOT, "AdcSR/testset/DIV2K-Val/HR/*.png")))
             + sorted(glob.glob(os.path.join(ROOT, "AdcSR/testset/DRealSR/HR/*.png")))
             + sorted(glob.glob(os.path.join(ROOT, "AdcSR/testset/RealSR/HR/*.png"))))
    if args.extra_data:
        for d in args.extra_data.split(","):
            paths += sorted(glob.glob(os.path.join(d.strip(), "*.png")))
    if not paths:
        raise RuntimeError("no training images found")
    rep = 20 if len(paths) < 8000 else max(1, 60000 // max(len(paths), 1))
    print(f"train images={len(paths)} repeat={rep}", flush=True)
    dl = DataLoader(PairDataset(paths, repeat=rep, mix=args.degrade_mix), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True)

    val_items = collect_crops(args.val_crops)
    ev = evaluate_model(model, val_items, M, device=DEV, dtype=DT, robust_beta=0.15)
    ev.pop("rows", None)
    print(f"[0] score={ev['score']:.4f} PSNR={ev['metrics']['psnr']:.3f} "
          f"lpips={ev['metrics']['lpips']:.4f} dists={ev['metrics']['dists']:.4f} "
          f"niqe={ev['metrics']['niqe']:.3f} musiq={ev['metrics']['musiq']:.2f} "
          f"maniqa={ev['metrics']['maniqa']:.4f} clipiqa={ev['metrics']['clipiqa']:.4f}", flush=True)

    best, hist, step, t0 = -1e9, [], 0, time.time()
    it = iter(dl)
    while step < args.steps:
        try:
            lq, gt = next(it)
        except StopIteration:
            it = iter(dl); continue
        step += 1
        lq, gt = lq.to(DEV, non_blocking=True), gt.to(DEV, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=DT):
            out, _ = model.forward_train(lq)
        out = out.float(); gt = gt.float(); lq = lq.float()
        # 以下全部 fp32：DISTS 在 autocast 下反传会全 nan（这个坑踩过一次）
        l1 = F.l1_loss(out, gt)
        lp = M["lpips"](out, gt).mean()
        ds = M["dists"](out, gt).mean()
        loss = args.w_l1 * l1 + args.w_lpips * lp + args.w_dists * ds
        loss = loss + args.w_fid * F.l1_loss(F.avg_pool2d(out, 16), F.avg_pool2d(lq, 16))
        if args.w_barrier > 0:
            bar = torch.zeros((), device=DEV)
            if args.lpips_cap > 0:
                bar = bar + F.relu(lp - args.lpips_cap) ** 2
            if args.dists_cap > 0:
                bar = bar + F.relu(ds - args.dists_cap) ** 2
            loss = loss + args.w_barrier * bar
        iq = out[:min(args.iqa_batch, out.shape[0])]
        which = step % 3
        if which == 0:
            r = 3.0 * args.w_clipiqa * M["clipiqa"](iq).mean()
        elif which == 1:
            r = 3.0 * args.w_maniqa * M["maniqa"](iq).mean()
        else:
            r = 3.0 * args.w_musiq * (M["musiq"](mosaic_in(iq, args.musiq_proxy)).mean() / 100.0)
        loss = loss - args.w_iqa * r
        if args.w_niqe > 0:
            loss = loss + args.w_niqe * M["niqe"](iq.clamp(0, 1)).mean() / 10.0

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        scaler.step(opt); scaler.update()

        if step % 25 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} l1={l1.item():.4f} "
                  f"lp={lp.item():.4f} dists={ds.item():.4f} iqa={r.item():.4f} "
                  f"{(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            ev = evaluate_model(model, val_items, M, device=DEV, dtype=DT, robust_beta=0.15)
            ev.pop("rows", None)
            m = ev["metrics"]
            hist.append({"step": step, **{k: float(v) for k, v in m.items()},
                         "score": float(ev["score"])})
            flag = ""
            if ev["score"] > best:
                best = ev["score"]; flag = " <- best"
                torch.save({"lora": {k: v.cpu() for k, v in model.tr.state_dict().items()
                                     if "lora_" in k},
                            "dec": ({k: v.cpu() for k, v in model.dec.state_dict().items()}
                                    if (args.train_decoder and model.dec is not None) else {}),
                            "step": step, "args": vars(args)},
                           os.path.join(args.out, "best.pth"))
            print(f"[val {step}] score={ev['score']:.4f} PSNR={m['psnr']:.3f} "
                  f"lpips={m['lpips']:.4f} dists={m['dists']:.4f} niqe={m['niqe']:.3f} "
                  f"musiq={m['musiq']:.2f} maniqa={m['maniqa']:.4f} "
                  f"clipiqa={m['clipiqa']:.4f}{flag}", flush=True)
            json.dump(hist, open(os.path.join(args.out, "log.json"), "w"), indent=2)
    print(f"done. best score={best:.4f}", flush=True)


main()
