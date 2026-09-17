"""教师候选 A：VOSR 多步采样能不能比单步更强。

rCM 压的是步数，所以教师必须是多步的；而我们本地只有单步蒸馏版 VOSR_0.5B_os
（args.json 里 distill_type='rcgm'，官方发布的就是同族方法蒸出来的）。
先看这个单步模型跑多步能不能涨——涨了就是最省事的教师，不涨就得去找别的多步基座。

同时扫 cfg_scale（零样本一直用的 0.5，从没调过）。
全部按官方口径：原生 4K tile 融合、3 张验证图六项。
"""
import os, sys, json, itertools, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
VOSR_DIR = "/cache/zihaohe/VOSR"
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
sys.path.insert(0, VOSR_DIR)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random
from PIL import Image
from torchvision.transforms import Normalize
from safetensors.torch import load_file
from settings import VAL
from metrics_utils import create_metrics

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
CK = "/cache/zihaohe/VOSR_weights/VOSR_0.5B_os"
A = json.load(open(os.path.join(CK, "args.json")))
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

from diffusers import AutoencoderKL
from models.light_decoder import LightDecoder
from models.lightningdit import LightningDiT

vae = AutoencoderKL.from_pretrained(os.path.join(VOSR_DIR, "preset/ckpts/stable-diffusion-2-1-base"),
                                    subfolder="vae", torch_dtype=DT).to(DEV).eval()
ck = torch.load(os.path.join(VOSR_DIR, "preset/ckpts/sd21_lwdecoder.pth"), map_location="cpu")
c = ck["config"]
dec = LightDecoder(in_channels=c["in_channels"], out_channels=c["out_channels"],
                   block_out_channels=tuple(c["block_out_channels"]),
                   layers_per_block=c["layers_per_block"])
dec.load_state_dict(ck["model_state_dict"]); dec = dec.to(DEV, dtype=DT).eval()
venc = torch.hub.load(os.path.join(VOSR_DIR, "preset/ckpts/torch_cache/facebookresearch_dinov2_main"),
                      "dinov2_vitb14", source="local", pretrained=False)
venc.load_state_dict(torch.load(
    os.path.join(VOSR_DIR, "preset/ckpts/torch_cache/checkpoints/dinov2_vitb14_pretrain.pth"),
    map_location="cpu"), strict=False)
if hasattr(venc, "head"): venc.head = nn.Identity()


def fwf(self, x, masks=None):
    feats = {}
    x = self.prepare_tokens_with_masks(x, masks)
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        feats[f"layer_{i}"] = x[:, 1:]
    return feats, self.norm(x)[:, 1:]


venc.forward_with_features = fwf.__get__(venc)
venc = venc.to(DEV, dtype=DT).eval()

dit = LightningDiT(
    input_size=A["resolution"] // 8, patch_size=A["patch_size"], in_channels=8, out_channels=4,
    hidden_size=A["dim"], depth=A["depth"], num_heads=A["num_heads"], mlp_ratio=A["mlp_ratio"],
    z_dims=A["enc_dim"], encdim_ratio=A["encdim_ratio"], auxiliary_time_cond=False,
    use_qknorm=A["use_qknorm"], use_swiglu=A["use_swiglu"], use_rope=A["use_rope"],
    use_rmsnorm=A["use_rmsnorm"], wo_shift=A["wo_shift"],
    num_fused_layers=len(A["layer_dinov2b_list"]))
w = load_file(os.path.join(CK, "checkpoints", "ema_model.safetensors"))
dit.load_state_dict({k: v.to(torch.float32) for k, v in w.items()}, strict=False)
dit = dit.to(DEV, dtype=DT).eval()
if hasattr(dit, "forward_flexible"):
    dit.forward = dit.forward_flexible
SF = vae.config.scaling_factor


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def cond_of(lq11):
    x = ((0.5 * lq11 + 0.5) * 255.) / 255.
    x = F.interpolate(x.float(), A["dinov2_size"], mode="bicubic").clip(0., 1.)
    x = Normalize(IMNET_MEAN, IMNET_STD)(x).to(DT)
    feats, x_norm = venc.forward_with_features(x)
    z = [v for k, v in feats.items() if k.startswith("layer_")]
    z[-1] = x_norm
    return [z[i] for i in A["layer_dinov2b_list"]]


@torch.no_grad()
def run_tile(lq01, n_steps, cfg):
    lq11 = lq01 * 2 - 1
    lat = vae.encode(lq11).latent_dist.mode() * SF
    cond = cond_of(lq11)
    g = torch.Generator(device=DEV).manual_seed(1234)
    z = torch.randn(lat.shape, generator=g, device=DEV, dtype=lat.dtype)
    B = lat.shape[0]
    t_seq = torch.linspace(1., 0., n_steps + 1, device=DEV)
    zero_cond = [torch.zeros_like(f) for f in cond]
    for i in range(n_steps):
        tc, tn = t_seq[i], t_seq[i + 1]
        # t_embedder 内部的正弦嵌入产 fp32，权重是 fp16，必须用 autocast 兜住
        with torch.autocast("cuda", dtype=DT):
            u = dit(torch.cat([lat, z], 1), tc.repeat(B).to(DT), None, cond)
            if cfg is not None and cfg != 1.0:
                u0 = dit(torch.cat([torch.zeros_like(lat), z], 1), tc.repeat(B).to(DT),
                         None, zero_cond)
                u = u0 + cfg * (u - u0)
        z = z - (tc - tn) * u.to(z.dtype)
    return ((dec(z / SF).clamp(-1, 1)) * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def tiled(x, n_steps, cfg, tile=512, ov=256):
    _, _, H, W = x.shape; st = tile - ov
    ys = list(range(0, max(H - tile, 0) + 1, st)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, st)) or [0]
    if ys[-1] + tile < H: ys.append(max(H - tile, 0))
    if xs[-1] + tile < W: xs.append(max(W - tile, 0))
    t = torch.linspace(-1, 1, tile, device=DEV); gg = torch.exp(-(t ** 2) / (2 * 0.45 ** 2))
    wm = (gg[:, None] * gg[None, :])[None, None].float()
    out = torch.zeros(1, 3, H, W, device=DEV); acc = torch.zeros(1, 1, H, W, device=DEV)
    for y in ys:
        for xx in xs:
            p = run_tile(x[:, :, y:y + tile, xx:xx + tile].to(DEV, dtype=DT), n_steps, cfg).float()
            out[:, :, y:y + tile, xx:xx + tile] += p * wm
            acc[:, :, y:y + tile, xx:xx + tile] += wm
    return (out / acc.clamp_min(1e-8)).clamp(0, 1)


M = create_metrics(DEV)
res = {}
GRID = [(1, None), (2, None), (4, None), (8, None), (1, 1.5), (4, 1.5), (4, 2.5)]
print(f"{'步数':>4}{'cfg':>6}{'LPIPS':>9}{'DISTS':>9}{'NIQE':>8}{'MUSIQ':>8}"
      f"{'MANIQA':>9}{'CLIPIQA':>9}{'PSNR':>8}{'Score':>9}", flush=True)
for n_steps, cfg in GRID:
    seed_all()
    acc = {k: [] for k in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")}
    ps = []
    try:
        for l, g in PAIRS:
            gt = load(os.path.join(VAL, g)).to(DEV)
            sr = tiled(load(os.path.join(VAL, l)), n_steps, cfg)
            for k in acc:
                acc[k].append(float(M[k](sr, gt) if k in ("lpips", "dists") else M[k](sr)))
            ps.append(10 * np.log10(1.0 / max(float(((sr - gt) ** 2).mean()), 1e-12)))
            del sr, gt; torch.cuda.empty_cache()
    except Exception as e:
        print(f"{n_steps:>4}{str(cfg):>6}  失败 {type(e).__name__}: {str(e)[:80]}", flush=True)
        continue
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    m["psnr"] = float(np.mean(ps))
    m["score"] = ((1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"] + m["maniqa"]
                  + m["musiq"] / 100.0 + max(0.0, (10 - m["niqe"]) / 10.0))
    res[f"n{n_steps}_cfg{cfg}"] = m
    print(f"{n_steps:>4}{str(cfg):>6}{m['lpips']:>9.4f}{m['dists']:>9.4f}{m['niqe']:>8.3f}"
          f"{m['musiq']:>8.2f}{m['maniqa']:>9.4f}{m['clipiqa']:>9.4f}{m['psnr']:>8.2f}"
          f"{m['score']:>9.4f}", flush=True)
    json.dump(res, open("/cache/zihaohe/csig_isr/eval_out/vosr_multistep.json", "w"),
              indent=2, ensure_ascii=False)
print("\n对照：VOSR零样本单步 3.5434 | 我们微调的 vosr_cap 4.1093 | TinySR run12 4.0025", flush=True)
