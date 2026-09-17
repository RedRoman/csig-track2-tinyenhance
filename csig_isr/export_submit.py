"""导出提交用 jit 模型（TinySR x1 + run8 权重）。

值域：官方 runner 的 infer 输入输出均为 [-1,1]，与本模型 VAE 的实际量纲一致，
     无需额外转换（训练路径 core(x01) 内部的 x01*2-1 与之等价）。
"""
import os, sys, json, gc
import torch
import torch.nn as nn

ROOT = "/cache/zihaohe/csig_isr"
TS = f"{ROOT}/TinySR"
sys.path.insert(0, TS)
os.chdir(TS)

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline

CKPT = f"{ROOT}/train_out/tinysr8_fid/best.pth"
OUT_DIR = f"{ROOT}/submit/model_dir"
os.makedirs(OUT_DIR, exist_ok=True)
dev, dt = torch.device("cuda"), torch.float16


class SubmitModel(nn.Module):
    """提交模型：输入输出均为 [-1,1] 值域的 [1,3,H,W]（与官方 runner 一致）。"""

    def __init__(self, vae, tr, ts, pooled):
        super().__init__()
        self.vae, self.tr = vae, tr
        self.register_buffer("ts", ts, persistent=False)
        self.register_buffer("pooled", pooled, persistent=False)

    def forward(self, x_m11):
        # 值域说明：训练/出图路径是 core(x01) 内部做 (x01*2-1) 后送 VAE.encode，
        # 即 VAE 实际收到的一直是 [-1,1]。官方 runner 传进来的已是 [-1,1]，
        # 故此处直接送入，不再做任何转换（已用真实图验证输出落在 [-1.22, 1.48]）。
        x = x_m11
        sf = self.vae.config.scaling_factor
        z = self.vae.encode(x).latents * sf
        pred = self.tr(hidden_states=z, timestep=self.ts,
                       pooled_projections=self.pooled, return_dict=False)[0]
        img = self.vae.decode((z - pred) / sf, return_dict=False)[0]
        return img.clamp(-1, 1)


print("1. 载入 TinySR + run8 权重")
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

print("2. warmup 触发惰性缓存（trace 前必须做，否则 trace 到残缺图）")
with torch.no_grad():
    w = torch.rand(1, 3, 512, 512, device=dev, dtype=dt) * 2 - 1
    zz = vae.encode(w).latents * vae.config.scaling_factor
    _ = tr(hidden_states=zz, timestep=ts, pooled_projections=pooled, return_dict=False)
assert getattr(tr, "initialized", False), "惰性缓存未触发"

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
cur = tr.state_dict(); hit = 0
for k, v in sd.get("lora", {}).items():
    if k in cur and cur[k].shape == v.shape:
        cur[k].copy_(v.to(cur[k].dtype)); hit += 1
print(f"   载入 {CKPT} (step {sd.get('step')})  LoRA {hit} 项")

print("3. merge LoRA 到主干（去 peft 包装，图更干净也更快）")
tr = tr.merge_and_unload()

model = SubmitModel(vae, tr, ts, pooled).to(dev).half().eval()
n_p = sum(p.numel() for p in model.parameters())
print(f"   参数量 {n_p/1e6:.2f} M")

# 用真实测试图做验证（随机噪声经 VAE 后趋于均值，值域窄，无法反映真实行为）
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
_a = np.asarray(Image.open(f"{ROOT}/data/赛题二/测试集/case1.jpg").convert("RGB"),
                dtype=np.float32) / 255.0
_t = torch.from_numpy(_a).permute(2, 0, 1)[None][:, :, 1000:1512, 1000:1512]
x = (_t * 2 - 1).to(dev, dtype=dt)
with torch.no_grad():
    ref = model(x).float().clone()
print(f"   eager 输出 {tuple(ref.shape)} 范围 [{ref.min():.3f}, {ref.max():.3f}]")


def bench(fn, n=60, warm=20):
    with torch.no_grad():
        for _ in range(warm):
            fn(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn(x)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / n


t_eager = bench(model)
print(f"   eager 时延 {t_eager:.3f} ms")

print("4. torch.jit.trace + freeze")
with torch.no_grad():
    traced = torch.jit.trace(model, x, check_trace=False).eval()
    frozen = None
    for tag, fn in [("freeze(optimize_numerics=False)",
                     lambda m: torch.jit.freeze(m, optimize_numerics=False)),
                    ("纯 trace", lambda m: m)]:
        try:
            frozen = fn(traced)
            print(f"   使用 {tag}")
            break
        except Exception as ex:
            print(f"   {tag} 失败: {str(ex)[:60]}")
    assert frozen is not None

path = os.path.join(OUT_DIR, "tinysr_x1.pt")
frozen.save(path)
print(f"   已存 {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

print("5. 独立加载验证（模拟主办方环境）")
del frozen, traced
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
loaded = torch.jit.load(path, map_location=dev)
loaded.eval()
with torch.no_grad():
    for _ in range(20):
        _ = loaded(x)
torch.cuda.synchronize()
with torch.no_grad():
    out = loaded(x).float()
d = (out - ref).abs()
rel = d.mean().item() / max(ref.abs().mean().item(), 1e-8)
t_jit = bench(loaded)
print(f"   与 eager 差异 max {d.max().item():.6f} mean {d.mean().item():.8f} 相对 {rel*100:.5f}%")
print(f"   jit 时延 {t_jit:.3f} ms  (eager {t_eager:.3f} ms, {t_eager/t_jit:.2f}x)")

with torch.no_grad():
    o1 = loaded(x).float().clone()
    _ = loaded(torch.rand(1, 3, 512, 512, device=dev, dtype=dt) * 2 - 1)
    o2 = loaded(x).float().clone()
print(f"   状态污染 max diff {(o1-o2).abs().max().item():.8f}")

print("6. 多分辨率（TinySR 的 pos_embed 硬编码为 1024 token = 512x512，")
print("   赛题时延只测 512x512，4K 出图也走 512 分块，故仅需 512 可用）")
for s_ in [512]:
    xt = torch.rand(1, 3, s_, s_, device=dev, dtype=dt) * 2 - 1
    try:
        with torch.no_grad():
            o = loaded(xt)
        print(f"   {s_}x{s_} -> {tuple(o.shape)} OK")
    except Exception as ex:
        print(f"   {s_}x{s_} 失败: {str(ex)[:50]}")

T_OSE = 212.768
sp = T_OSE / t_jit
print()
print(f"加速比（OSEDiff 基准 {T_OSE} ms）: {sp:.2f}x   Speedup^0.2 = {sp**0.2:.4f}")
json.dump({"t_eager": t_eager, "t_jit": t_jit, "speedup": sp,
           "rel_diff": rel, "params_M": n_p / 1e6},
          open(f"{ROOT}/eval_out/submit_model.json", "w"), indent=1)
print(f"已存 eval_out/submit_model.json")
