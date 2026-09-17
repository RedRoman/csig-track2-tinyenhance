"""TinySR x1 的 torch.jit 导出 + 时延实测。

关键：TinySR 有惰性缓存（首次 forward 会 del time_text_embed/norm_out/各block norm1），
必须先 warmup 再 trace，否则会 trace 到残缺图。
"""
import os, sys, argparse, json
import numpy as np
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
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, default=f"{ROOT}/train_out/tinysr2/best.pth")
ap.add_argument("--out", type=str, default=f"{ROOT}/x1_model/tinysr_x1.pt")
args = ap.parse_args()


class TinySRx1(nn.Module):
    def __init__(self, vae, tr, ts, pooled):
        super().__init__()
        self.vae, self.tr = vae, tr
        self.register_buffer("ts", ts, persistent=False)
        self.register_buffer("pooled", pooled, persistent=False)
    def forward(self, lq01):
        x = lq01 * 2 - 1
        sf = self.vae.config.scaling_factor
        z = self.vae.encode(x).latents * sf
        pred = self.tr(hidden_states=z, timestep=self.ts,
                       pooled_projections=self.pooled, return_dict=False)[0]
        img = self.vae.decode((z - pred) / sf, return_dict=False)[0]
        return (img * 0.5 + 0.5).clamp(0, 1)


print("1. 载入模型")
tr = TinySD3Transformer2DModel.from_pretrained(
    f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr", subfolder="transformer",
    torch_dtype=dt, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
vae = AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable", torch_dtype=dt)
lc = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian",
                target_modules=["to_k","to_q","to_v","to_out.0","proj","linear","linear_1","linear_2","net.2"])
tr.add_adapter(lc); tr.enable_adapters()
load_lora_state_dict(StableDiffusion3Pipeline.lora_state_dict(
    f"{TS}/checkpoint/tinysr", weight_name="transformer.safetensors"), tr)
vae = vae.to(dev, dtype=dt).eval(); tr = tr.to(dev, dtype=dt).eval()
ts = torch.tensor([1000.0], device=dev, dtype=dt)
pooled = torch.load(f"{TS}/dataset/default/pool_embeds.pt", map_location=dev).to(dtype=dt)

print("2. warmup 触发惰性缓存（trace 前必须做）")
with torch.no_grad():
    w = torch.rand(1,3,512,512, device=dev, dtype=dt)*2-1
    zz = vae.encode(w).latents*vae.config.scaling_factor
    _ = tr(hidden_states=zz, timestep=ts, pooled_projections=pooled, return_dict=False)
assert getattr(tr,"initialized",False), "缓存未触发"
print("   缓存已触发")

if args.ckpt and os.path.exists(args.ckpt):
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cur = tr.state_dict(); hit = 0
    for k,v in sd["lora"].items():
        if k in cur and cur[k].shape == v.shape:
            cur[k].copy_(v.to(cur[k].dtype)); hit += 1
    print(f"3. 载入训练权重 {args.ckpt} (step {sd.get('step')}) 匹配 {hit}/{len(sd['lora'])}")

# LoRA merge 到主干（去掉 peft 包装，trace 更干净也更快）
try:
    tr = tr.merge_and_unload() if hasattr(tr,"merge_and_unload") else tr
    print("4. LoRA 已 merge")
except Exception as e:
    print(f"4. merge 跳过: {str(e)[:60]}")

model = TinySRx1(vae, tr, ts, pooled).to(dev).half().eval()

def bench(fn, inp, n=60, warm=15):
    with torch.no_grad():
        for _ in range(warm): fn(inp)
        torch.cuda.synchronize()
        s,e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n): fn(inp)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

x = torch.rand(1,3,512,512, device=dev, dtype=dt)
with torch.no_grad():
    ref = model(x).float().clone()
t_eager = bench(model, x)
print(f"\n5. eager 时延 {t_eager:.3f} ms")

print("6. trace（freeze 在此模型上触发 PyTorch conv-folding 断言 bug，改用 optimize_for_inference 或纯 trace）")
with torch.no_grad():
    traced = torch.jit.trace(model, x, check_trace=False)
    traced = traced.eval()
    frozen = None
    for name, fn_opt in [("freeze(optimize_numerics=False)",
                          lambda m: torch.jit.freeze(m, optimize_numerics=False)),
                         ("纯 trace", lambda m: m)]:
        try:
            frozen = fn_opt(traced)
            print(f"   使用 {name}")
            break
        except Exception as ex:
            print(f"   {name} 失败: {str(ex)[:70]}")
    assert frozen is not None
os.makedirs(os.path.dirname(args.out), exist_ok=True)
frozen.save(args.out)
print(f"   已存 {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

del frozen
try:
    del traced
except NameError:
    pass
import gc
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
loaded = torch.jit.load(args.out, map_location=dev); loaded.eval()
# 充分预热：jit 首次运行有编译开销，预热不足会得到严重偏高的假数字
with torch.no_grad():
    for _ in range(20): _ = loaded(x)
torch.cuda.synchronize()
    out = loaded(x).float()
d = (out-ref).abs()
rel = d.mean().item()/max(ref.abs().mean().item(),1e-8)
t_jit = bench(loaded, x)

print(f"\n7. 验证")
print(f"   与 eager 差异 max {d.max().item():.6f}  mean {d.mean().item():.8f}  相对 {rel*100:.5f}%")
print(f"   jit 时延 {t_jit:.3f} ms  （eager {t_eager:.3f} ms，提速 {t_eager/t_jit:.2f}x）")

# 多次调用一致性
with torch.no_grad():
    o1 = loaded(x).float().clone()
    _ = loaded(torch.rand(1,3,512,512,device=dev,dtype=dt))
    o2 = loaded(x).float().clone()
print(f"   状态污染检查 max diff {(o1-o2).abs().max().item():.8f}")

T_OSE = 212.768
print(f"\n8. 加速比（OSEDiff 基准 {T_OSE} ms）")
for tag,t in [("eager",t_eager),("jit frozen",t_jit)]:
    sp = T_OSE/t
    print(f"   {tag:<12}{t:>8.2f} ms  {sp:>7.2f}x  Sp^0.2={sp**0.2:.4f}")
json.dump({"t_eager":t_eager,"t_jit":t_jit,"rel_diff":rel},
          open(f"{ROOT}/eval_out/tinysr_jit.json","w"), indent=1)
