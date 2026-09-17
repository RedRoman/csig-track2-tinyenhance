"""导出三种方法的对比图，供肉眼检查是否有伪纹理/过锐化。"""
import os, sys
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
OUT = f"{ROOT}/visualizations/x1_compare"
os.makedirs(OUT, exist_ok=True)
VAL = f"{ROOT}/data/赛题二/验证集"
dev, dt = torch.device("cuda"), torch.float16


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def save(t, p):
    a = (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(a).save(p)


# 固定挑三个纹理丰富的 crop
PICKS = [("case1_lq.jpg", "case1_gt.png", 101), ("case2_lq.jpg", "case2_gt.png", 202),
         ("case3_lq.jpg", "case3_gt.jpg", 303)]
crops = []
for lqn, gtn, seed in PICKS:
    lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
    if lq_f.shape != gt_f.shape:
        continue
    rng = np.random.RandomState(seed)
    _, _, H, W = gt_f.shape
    got, tries = 0, 0
    while got < 2 and tries < 40:
        tries += 1
        y, x = rng.randint(0, H-512), rng.randint(0, W-512)
        gt = gt_f[:, :, y:y+512, x:x+512]
        if gt.std().item() < 0.06:
            continue
        got += 1
        crops.append((lqn[:5], got, lq_f[:, :, y:y+512, x:x+512].clone(), gt.clone()))
print(f"选取 {len(crops)} 个 crop")

# TinySR
sys.path.insert(0, f"{ROOT}/TinySR"); os.chdir(f"{ROOT}/TinySR")
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
TS = f"{ROOT}/TinySR"
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
with torch.no_grad():
    w = torch.rand(1,3,512,512, device=dev, dtype=dt)*2-1
    zz = vae.encode(w).latents * vae.config.scaling_factor
    _ = tr(hidden_states=zz, timestep=ts, pooled_projections=pooled, return_dict=False)


@torch.no_grad()
def tinysr_infer(lq01):
    x = lq01.to(dev, dtype=dt)*2-1
    z = vae.encode(x).latents * vae.config.scaling_factor
    p = tr(hidden_states=z, timestep=ts, pooled_projections=pooled, return_dict=False)[0]
    img = vae.decode((z-p)/vae.config.scaling_factor, return_dict=False)[0]
    return (img.float()*0.5+0.5).clamp(0,1)


for tag, i, lq, gt in crops:
    save(lq, f"{OUT}/{tag}_{i}_1_LQ.png")
    save(gt, f"{OUT}/{tag}_{i}_2_GT.png")
    save(tinysr_infer(lq), f"{OUT}/{tag}_{i}_3_TinySR.png")
print(f"已导出到 {OUT}")
