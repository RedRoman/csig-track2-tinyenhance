"""排查 TinySR jit 变慢 2.9x 的原因，找出可用的导出方式。

对比：eager / trace未freeze / freeze(optimize_numerics=False) / optimize_for_inference
并分段测（VAE enc / DiT / VAE dec）定位是哪块退化。
"""
import os, sys, json
import torch
import torch.nn as nn

ROOT = "/cache/zihaohe/csig_isr"
TS = f"{ROOT}/TinySR"
sys.path.insert(0, TS); os.chdir(TS)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline

dev, dt = torch.device("cuda"), torch.float16


class TinySRx1(nn.Module):
    def __init__(s, vae, tr, ts, pooled):
        super().__init__(); s.vae, s.tr = vae, tr
        s.register_buffer("ts", ts, persistent=False)
        s.register_buffer("pooled", pooled, persistent=False)
    def forward(s, lq01):
        x = lq01*2-1; sf = s.vae.config.scaling_factor
        z = s.vae.encode(x).latents*sf
        p = s.tr(hidden_states=z, timestep=s.ts, pooled_projections=s.pooled, return_dict=False)[0]
        return (s.vae.decode((z-p)/sf, return_dict=False)[0]*0.5+0.5).clamp(0,1)


tr = TinySD3Transformer2DModel.from_pretrained(
    f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr", subfolder="transformer",
    torch_dtype=dt, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
vae = AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable", torch_dtype=dt)
lc = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian",
                target_modules=["to_k","to_q","to_v","to_out.0","proj","linear","linear_1","linear_2","net.2"])
tr.add_adapter(lc); tr.enable_adapters()
load_lora_state_dict(StableDiffusion3Pipeline.lora_state_dict(
    f"{TS}/checkpoint/tinysr", weight_name="transformer.safetensors"), tr)
vae = vae.to(dev,dtype=dt).eval(); tr = tr.to(dev,dtype=dt).eval()
ts = torch.tensor([1000.0], device=dev, dtype=dt)
pooled = torch.load(f"{TS}/dataset/default/pool_embeds.pt", map_location=dev).to(dtype=dt)
with torch.no_grad():
    w = torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
    zz = vae.encode(w).latents*vae.config.scaling_factor
    _ = tr(hidden_states=zz, timestep=ts, pooled_projections=pooled, return_dict=False)
sd = torch.load(f"{ROOT}/train_out/tinysr2/best.pth", map_location="cpu", weights_only=False)
cur = tr.state_dict()
for k,v in sd["lora"].items():
    if k in cur and cur[k].shape==v.shape: cur[k].copy_(v.to(cur[k].dtype))
tr = tr.merge_and_unload()
model = TinySRx1(vae, tr, ts, pooled).to(dev).half().eval()

x = torch.rand(1,3,512,512, device=dev, dtype=dt)

def bench(fn, n=50, warm=15):
    with torch.no_grad():
        for _ in range(warm): fn(x)
        torch.cuda.synchronize()
        s,e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n): fn(x)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

print("=" * 66)
print("A. 各导出方式对比")
print("=" * 66)
t_eager = bench(model)
print(f"  {'eager':<38}{t_eager:>9.2f} ms   1.00x")

with torch.no_grad():
    traced = torch.jit.trace(model, x, check_trace=False).eval()
t_tr = bench(traced)
print(f"  {'trace (未 freeze)':<38}{t_tr:>9.2f} ms   {t_eager/t_tr:>4.2f}x")

try:
    fz0 = torch.jit.freeze(traced, optimize_numerics=False)
    with torch.no_grad():
        for _ in range(5): fz0(x)
    t_f0 = bench(fz0)
    print(f"  {'freeze(optimize_numerics=False)':<38}{t_f0:>9.2f} ms   {t_eager/t_f0:>4.2f}x")
except Exception as ex:
    print(f"  freeze(False) 失败: {str(ex)[:50]}")

try:
    ofi = torch.jit.optimize_for_inference(traced)
    with torch.no_grad():
        for _ in range(5): ofi(x)
    t_ofi = bench(ofi)
    print(f"  {'optimize_for_inference':<38}{t_ofi:>9.2f} ms   {t_eager/t_ofi:>4.2f}x")
except Exception as ex:
    print(f"  optimize_for_inference 失败: {str(ex)[:50]}")

print()
print("=" * 66)
print("B. 分段定位（哪块被 jit 拖慢）")
print("=" * 66)
with torch.no_grad():
    z = vae.encode(x*2-1).latents*vae.config.scaling_factor
    p = tr(hidden_states=z, timestep=ts, pooled_projections=pooled, return_dict=False)[0]
    zo = (z-p)/vae.config.scaling_factor

def b2(fn, inp, n=50, warm=15):
    with torch.no_grad():
        for _ in range(warm): fn(inp)
        torch.cuda.synchronize()
        s,e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n): fn(inp)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

class EncW(nn.Module):
    def __init__(s,v): super().__init__(); s.v=v
    def forward(s,a): return s.v.encode(a).latents
class DecW(nn.Module):
    def __init__(s,v): super().__init__(); s.v=v
    def forward(s,a): return s.v.decode(a, return_dict=False)[0]
class TrW(nn.Module):
    def __init__(s,t,ts,pl): super().__init__(); s.t=t; s.register_buffer("ts",ts,persistent=False); s.register_buffer("pl",pl,persistent=False)
    def forward(s,a): return s.t(hidden_states=a, timestep=s.ts, pooled_projections=s.pl, return_dict=False)[0]

enc, dec, trw = EncW(vae).eval(), DecW(vae).eval(), TrW(tr,ts,pooled).eval()
xin = x*2-1
print(f"  {'模块':<16}{'eager':>10}{'jit':>10}{'比值':>8}")
for nm, mod, inp in [("VAE encoder", enc, xin), ("DiT", trw, z), ("VAE decoder", dec, zo)]:
    te = b2(mod, inp)
    with torch.no_grad():
        j = torch.jit.trace(mod, inp, check_trace=False).eval()
        try: j = torch.jit.freeze(j, optimize_numerics=False)
        except Exception: pass
        for _ in range(5): j(inp)
    tj = b2(j, inp)
    print(f"  {nm:<16}{te:>10.3f}{tj:>10.3f}{te/tj:>8.2f}x")
