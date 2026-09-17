"""4K 测试集全量出图 + 无参考指标统计 + 视觉抽检拼图。

赛题要点：时延只测 512x512 单次前向，4K 分块策略【不计入时延】，
所以这里用大 overlap + 高斯权重融合追求质量，不必吝惜计算。

输出：
  output_dir/case1.jpg ... case100.jpg   （提交用，jpg 格式，与测试图同名）
  visualizations/test_4k/                 抽检对比图
"""
import os, sys, json, time, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
TS = f"{ROOT}/TinySR"
sys.path.insert(0, TS)
os.chdir(TS)

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from utils.wavelet_color_fix import wavelet_color_fix
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
from torchvision.transforms import ToPILImage, ToTensor

TEST = f"{ROOT}/data/赛题二/测试集"
dev, dt = torch.device("cuda"), torch.float16

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, default=f"{ROOT}/train_out/tinysr8_fid/best.pth")
ap.add_argument("--out", type=str, default=f"{ROOT}/submit/output_dir")
ap.add_argument("--tile", type=int, default=512)
ap.add_argument("--overlap", type=int, default=256, help="出图用大 overlap（不计入时延）")
ap.add_argument("--colorfix", type=str, default="wavelet", choices=["wavelet", "none"])
ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全部）")
ap.add_argument("--quality", type=int, default=97)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def build():
    tr = TinySD3Transformer2DModel.from_pretrained(
        f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr", subfolder="transformer",
        torch_dtype=dt, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    vae = AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable", torch_dtype=dt)
    lc = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj",
                                    "linear", "linear_1", "linear_2", "net.2"])
    tr.add_adapter(lc)
    tr.enable_adapters()
    load_lora_state_dict(StableDiffusion3Pipeline.lora_state_dict(
        f"{TS}/checkpoint/tinysr", weight_name="transformer.safetensors"), tr)
    vae = vae.to(dev, dtype=dt).eval()
    tr = tr.to(dev, dtype=dt).eval()
    ts = torch.tensor([1000.0], device=dev, dtype=dt)
    pooled = torch.load(f"{TS}/dataset/default/pool_embeds.pt", map_location=dev).to(dtype=dt)
    # warmup 触发惰性缓存
    with torch.no_grad():
        w = torch.rand(1, 3, 512, 512, device=dev, dtype=dt) * 2 - 1
        zz = vae.encode(w).latents * vae.config.scaling_factor
        _ = tr(hidden_states=zz, timestep=ts, pooled_projections=pooled, return_dict=False)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cur = tr.state_dict(); hit = 0
    for k, v in sd.get("lora", {}).items():
        if k in cur and cur[k].shape == v.shape:
            cur[k].copy_(v.to(cur[k].dtype)); hit += 1
    print(f"载入 {args.ckpt} (step {sd.get('step')})  LoRA {hit} 项", flush=True)
    return vae, tr, ts, pooled


vae, tr, ts, pooled = build()


def blend_w(h, w, overlap, sigma=0.45):
    """高斯融合权重。

    tile 策略扫描（16 组配置，见 eval_out/tile_sweep.json）结论：
      高斯 > Hann（所有 overlap 档位）
      不加镜像 padding > 加（所有档位）
      overlap 256 最优（Score 3.9309 vs ov128 的 3.9187）
    先前改 Hann 是为了消"接缝"，但那个接缝后来证实是检测指标的假阳性。
    """
    yy = torch.linspace(-1, 1, h, device=dev)
    xx = torch.linspace(-1, 1, w, device=dev)
    gy = torch.exp(-(yy ** 2) / (2 * sigma ** 2))
    gx = torch.exp(-(xx ** 2) / (2 * sigma ** 2))
    return (gy[:, None] * gx[None, :])[None, None].float()


@torch.no_grad()
def core(x01):
    """512 tile 的核心推理（与测速路径完全一致）。"""
    x = x01 * 2 - 1
    sf = vae.config.scaling_factor
    z = vae.encode(x).latents * sf
    p = tr(hidden_states=z, timestep=ts, pooled_projections=pooled, return_dict=False)[0]
    img = vae.decode((z - p) / sf, return_dict=False)[0]
    return (img.float() * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def tiled(img01, tile, overlap):
    """分块推理。镜像 padding 保证边界 tile 有完整上下文，Hann 窗融合消除接缝。"""
    _, _, H0, W0 = img01.shape
    pad = 0          # 扫描证明镜像 padding 在所有档位都是负作用
    _, _, H, W = img01.shape
    stride = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < H:
        ys.append(max(H - tile, 0))
    if not xs or xs[-1] + tile < W:
        xs.append(max(W - tile, 0))
    out = torch.zeros(1, 3, H, W, device=dev, dtype=torch.float32)
    acc = torch.zeros(1, 1, H, W, device=dev, dtype=torch.float32)
    wm = blend_w(tile, tile, overlap)
    for y in ys:
        for x in xs:
            patch = img01[:, :, y:y + tile, x:x + tile].to(dev, dtype=dt)
            o = core(patch)
            out[:, :, y:y + tile, x:x + tile] += o * wm
            acc[:, :, y:y + tile, x:x + tile] += wm
    return (out / acc.clamp(min=1e-8)).clamp(0, 1)


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


files = sorted(glob.glob(f"{TEST}/*.jpg"),
               key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
if args.limit:
    files = files[:args.limit]
print(f"待处理 {len(files)} 张，tile={args.tile} overlap={args.overlap} colorfix={args.colorfix}",
      flush=True)

import pyiqa
NR = {"niqe": pyiqa.create_metric("niqe", device=dev),
      "musiq": pyiqa.create_metric("musiq", device=dev),
      "maniqa": pyiqa.create_metric("maniqa-pipal", device=dev),
      "clipiqa": pyiqa.create_metric("clipiqa", device=dev)}

stats = {"lq": {k: [] for k in NR}, "sr": {k: [] for k in NR}}
t0 = time.time()
for i, f in enumerate(files, 1):
    name = os.path.basename(f)
    lq = load_img(f)
    sr = tiled(lq, args.tile, args.overlap)
    if args.colorfix == "wavelet":
        fx = wavelet_color_fix(ToPILImage()(sr[0].cpu()), ToPILImage()(lq[0].cpu()))
        sr = ToTensor()(fx)[None].to(dev)
    arr = (sr[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(args.out, name), quality=args.quality,
                              subsampling=0)
    # 中心 crop 抽样统计无参考指标（全图算太慢）
    _, _, H, W = sr.shape
    cs, ce = H // 2 - 256, H // 2 + 256
    ws, we = W // 2 - 256, W // 2 + 256
    for tag, im in [("lq", lq.to(dev)), ("sr", sr)]:
        c = im[:, :, cs:ce, ws:we]
        for k, fn in NR.items():
            stats[tag][k].append(fn(c).item())
    del lq, sr
    torch.cuda.empty_cache()
    if i % 10 == 0 or i == len(files):
        el = time.time() - t0
        print(f"  [{i}/{len(files)}] {name}  {el/i:.2f}s/张  预计剩余 {(len(files)-i)*el/i/60:.1f} min",
              flush=True)

print()
print("=" * 72)
print("测试集无参考指标（中心 512 crop 抽样，100 张平均）")
print("=" * 72)
res = {}
for tag in ["lq", "sr"]:
    m = {k: float(np.mean(v)) for k, v in stats[tag].items()}
    res[tag] = m
print(f"  {'':<8}{'NIQE↓':>10}{'MUSIQ↑':>10}{'MANIQA↑':>10}{'CLIPIQA↑':>10}{'无参考4项贡献':>14}")
for tag, lab in [("lq", "LQ 输入"), ("sr", "我们输出")]:
    m = res[tag]
    c = m["clipiqa"] + m["maniqa"] + m["musiq"] / 100 + max(0, (10 - m["niqe"]) / 10)
    print(f"  {lab:<8}{m['niqe']:>10.4f}{m['musiq']:>10.2f}{m['maniqa']:>10.4f}"
          f"{m['clipiqa']:>10.4f}{c:>14.4f}")
    res[tag]["nr_contrib"] = c
print(f"\n  提升 {res['sr']['nr_contrib'] - res['lq']['nr_contrib']:+.4f}")
json.dump(res, open(f"{ROOT}/eval_out/test4k_nr.json", "w"), indent=1)
print(f"\n输出目录 {args.out}")
print(f"总耗时 {(time.time()-t0)/60:.1f} min")
