"""TinySR x1 冒烟测试 + 时延 + 尺寸流验证。

TinySR 原生流程（x4）：
    LR 128 -> bicubic 上采到 512 -> VAE.encode -> 64x64x16 latent
    -> DiT 预测残差 -> z_out = z_in - z_pred -> VAE.decode -> 512

x1 改造：输入本来就是 512，直接跳过 bicubic 那步，其余一行不改。
关键优势：残差预测（z_in - z_pred）天然契合"低频无损、只补高频"的任务。
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/cache/zihaohe/csig_isr"
TS = f"{ROOT}/TinySR"
sys.path.insert(0, TS)
os.chdir(TS)

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline

BACKBONE = f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr"
VAE = f"{TS}/checkpoint/vae/separable"
LORA = f"{TS}/checkpoint/tinysr"
EMB = f"{TS}/dataset/default"
dev = torch.device("cuda")
dt = torch.float16


print("=" * 74)
print("1. 载入 TinySR")
print("=" * 74)
tr = TinySD3Transformer2DModel.from_pretrained(
    BACKBONE, subfolder="transformer", torch_dtype=dt,
    low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
print(f"   transformer 载入完成")

vae = AutoencoderTiny.from_pretrained(VAE, torch_dtype=dt)
print(f"   VAE(AutoencoderTiny) 载入完成  latent_channels={vae.config.latent_channels}")

cfg = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian",
                 target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj",
                                 "linear", "linear_1", "linear_2", "net.2"])
tr.add_adapter(cfg)
tr.enable_adapters()
lsd = StableDiffusion3Pipeline.lora_state_dict(LORA, weight_name="transformer.safetensors")
load_lora_state_dict(lsd, tr)
print(f"   LoRA 载入完成 ({len(lsd)} 个张量)")

vae = vae.to(dev, dtype=dt).eval()
tr = tr.to(dev, dtype=dt).eval()

n_tr = sum(p.numel() for p in tr.parameters())
n_blk = sum(p.numel() for p in tr.transformer_blocks.parameters())
n_vae = sum(p.numel() for p in vae.parameters())
print(f"   transformer 总参数 {n_tr/1e6:.2f} M (blocks {n_blk/1e6:.2f} M)")
print(f"   VAE 参数 {n_vae/1e6:.2f} M")
print(f"   合计 {(n_tr+n_vae)/1e6:.2f} M")

timesteps = torch.tensor([1000.0], device=dev, dtype=dt)
pooled = torch.load(os.path.join(EMB, "pool_embeds.pt"), map_location=dev).to(dtype=dt)
print(f"   预缓存 pooled_prompt_embeds {tuple(pooled.shape)}（无需 text encoder）")

print()
print("=" * 74)
print("2. x1 尺寸流验证（512 -> 512，跳过 bicubic 上采样）")
print("=" * 74)
x = torch.rand(1, 3, 512, 512, device=dev, dtype=dt) * 2 - 1
with torch.no_grad():
    # warmup：TinySR 首次 forward 会做惰性缓存（del self.norm_out 等）
    z = vae.encode(x).latents * vae.config.scaling_factor
    print(f"   VAE.encode   {tuple(x.shape)} -> {tuple(z.shape)}")
    pred = tr(hidden_states=z, timestep=timesteps, pooled_projections=pooled,
              return_dict=False)[0]
    print(f"   DiT 预测残差  {tuple(z.shape)} -> {tuple(pred.shape)}")
    z_out = z - pred
    img = vae.decode(z_out / vae.config.scaling_factor, return_dict=False)[0]
    print(f"   VAE.decode   {tuple(z_out.shape)} -> {tuple(img.shape)}")
    ok = img.shape == x.shape
    print(f"   端到端 {tuple(x.shape)} -> {tuple(img.shape)}  {'尺寸正确' if ok else '尺寸错误'}")
    print(f"   输出范围 [{img.min().item():.3f}, {img.max().item():.3f}]")

print()
print("=" * 74)
print("3. 时延（512x512, fp16, 赛题口径）")
print("=" * 74)


def bench(fn, n=60, warm=15):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / n


with torch.no_grad():
    z = vae.encode(x).latents * vae.config.scaling_factor
    pred = tr(hidden_states=z, timestep=timesteps, pooled_projections=pooled,
              return_dict=False)[0]
    z_out = z - pred

t_enc = bench(lambda: vae.encode(x))
t_dit = bench(lambda: tr(hidden_states=z, timestep=timesteps,
                         pooled_projections=pooled, return_dict=False))
t_dec = bench(lambda: vae.decode(z_out / vae.config.scaling_factor, return_dict=False))


def full():
    zz = vae.encode(x).latents * vae.config.scaling_factor
    pp = tr(hidden_states=zz, timestep=timesteps, pooled_projections=pooled,
            return_dict=False)[0]
    return vae.decode((zz - pp) / vae.config.scaling_factor, return_dict=False)


t_all = bench(full, 40, 10)

print(f"   {'模块':<26}{'ms':>10}{'占比':>9}")
for nm, t in [("VAE encoder", t_enc), ("DiT (12层 MMDiT)", t_dit), ("VAE decoder", t_dec)]:
    print(f"   {nm:<26}{t:>10.3f}{t/t_all*100:>8.1f}%")
print(f"   {'-'*44}")
print(f"   {'端到端':<26}{t_all:>10.3f}{100.0:>8.1f}%")

print()
print("=" * 74)
print("4. 与 Samsung x1 对比")
print("=" * 74)
print(f"   {'模型':<28}{'参数量':>12}{'时延ms':>11}")
print(f"   {'Samsung x1 (eager)':<28}{'199.96 M':>12}{18.80:>11.2f}")
print(f"   {'Samsung x1 (jit frozen)':<28}{'199.96 M':>12}{14.16:>11.2f}")
print(f"   {'TinySR x1 (eager)':<28}{f'{(n_tr+n_vae)/1e6:.2f} M':>12}{t_all:>11.2f}")
