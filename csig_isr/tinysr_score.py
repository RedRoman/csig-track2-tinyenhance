"""TinySR x1 zero-shot Score（未针对 x1 训练，作为对照）。"""
import os, sys, json
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
TS = f"{ROOT}/TinySR"
sys.path.insert(0, TS)
os.chdir(TS)

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline

VAL = f"{ROOT}/data/赛题二/验证集"
dev, dt = torch.device("cuda"), torch.float16
PAIRS = [("case1_lq.jpg", "case1_gt.png", 101),
         ("case2_lq.jpg", "case2_gt.png", 202),
         ("case3_lq.jpg", "case3_gt.jpg", 303)]


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


tr = TinySD3Transformer2DModel.from_pretrained(
    f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr", subfolder="transformer",
    torch_dtype=dt, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
vae = AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable", torch_dtype=dt)
cfg = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian",
                 target_modules=["to_k","to_q","to_v","to_out.0","proj","linear","linear_1","linear_2","net.2"])
tr.add_adapter(cfg); tr.enable_adapters()
load_lora_state_dict(StableDiffusion3Pipeline.lora_state_dict(
    f"{TS}/checkpoint/tinysr", weight_name="transformer.safetensors"), tr)
vae = vae.to(dev, dtype=dt).eval(); tr = tr.to(dev, dtype=dt).eval()
ts = torch.tensor([1000.0], device=dev, dtype=dt)
pooled = torch.load(f"{TS}/dataset/default/pool_embeds.pt", map_location=dev).to(dtype=dt)

# warmup（触发惰性缓存，之后行为才稳定）
with torch.no_grad():
    w = torch.rand(1, 3, 512, 512, device=dev, dtype=dt) * 2 - 1
    zz = vae.encode(w).latents * vae.config.scaling_factor
    _ = tr(hidden_states=zz, timestep=ts, pooled_projections=pooled, return_dict=False)


@torch.no_grad()
def infer(lq01):
    x = lq01.to(dev, dtype=dt) * 2 - 1
    z = vae.encode(x).latents * vae.config.scaling_factor
    p = tr(hidden_states=z, timestep=ts, pooled_projections=pooled, return_dict=False)[0]
    img = vae.decode((z - p) / vae.config.scaling_factor, return_dict=False)[0]
    return (img.float() * 0.5 + 0.5).clamp(0, 1)


crops = []
for lqn, gtn, seed in PAIRS:
    lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
    if lq_f.shape != gt_f.shape:
        continue
    rng = np.random.RandomState(seed)
    _, _, H, W = gt_f.shape
    got, tries = 0, 0
    while got < 6 and tries < 72:
        tries += 1
        y, x = rng.randint(0, H-512), rng.randint(0, W-512)
        gt = gt_f[:, :, y:y+512, x:x+512]
        if gt.std().item() < 0.045:
            continue
        got += 1
        crops.append((lq_f[:, :, y:y+512, x:x+512].clone(), gt.clone()))
print(f"验证 crop {len(crops)}")

import pyiqa
M = {"lpips": pyiqa.create_metric("lpips", device=dev),
     "dists": pyiqa.create_metric("dists", device=dev),
     "niqe": pyiqa.create_metric("niqe", device=dev),
     "musiq": pyiqa.create_metric("musiq", device=dev),
     "maniqa": pyiqa.create_metric("maniqa-pipal", device=dev),
     "clipiqa": pyiqa.create_metric("clipiqa", device=dev),
     "psnr": pyiqa.create_metric("psnr", device=dev, test_y_channel=True, color_space="ycbcr")}

res = {k: [] for k in M}
for lq, gt in crops:
    out = infer(lq)
    gt = gt.to(dev)
    for k, fn in M.items():
        v = fn(out, gt) if k in ("lpips", "dists", "psnr") else fn(out)
        res[k].append(v.item())
    torch.cuda.empty_cache()

m = {k: float(np.mean(v)) for k, v in res.items()}
score = ((1-m["lpips"]) + (1-m["dists"]) + m["clipiqa"] + m["maniqa"]
         + m["musiq"]/100 + max(0, (10-m["niqe"])/10))
print()
print("=" * 56)
print("  TinySR x1 zero-shot（未针对 x1 训练）")
print("=" * 56)
for k in ["lpips","dists","clipiqa","maniqa","musiq","niqe","psnr"]:
    print(f"  {k:<12}{m[k]:>12.4f}")
print(f"  {'Score':<12}{score:>12.4f}")
json.dump({"metrics": m, "score": score},
          open(f"{ROOT}/eval_out/score_tinysr_zeroshot.json", "w"), indent=1)
