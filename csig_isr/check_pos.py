"""确认 TinySR 的 pos_embed 是否正确加载（from_pretrained 报了 shape mismatch）。"""
import os, sys, torch
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0,TS); os.chdir(TS)

# 1. 缓存文件本身
pe = torch.load(f"{TS}/dataset/cache/pos_embed.pt", map_location="cpu")
print(f"dataset/cache/pos_embed.pt  shape={tuple(pe.shape)}  dtype={pe.dtype}")
print(f"  统计 mean={pe.float().mean():.6f} std={pe.float().std():.6f}")

# 2. checkpoint 里的
from safetensors import safe_open
p=f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr/transformer/diffusion_pytorch_model.safetensors"
with safe_open(p, framework="pt") as f:
    keys=[k for k in f.keys() if "pos_embed" in k]
    print(f"\ncheckpoint 中 pos_embed 相关键: {keys}")
    for k in keys:
        t=f.get_tensor(k)
        print(f"  {k}  shape={tuple(t.shape)}  mean={t.float().mean():.6f} std={t.float().std():.6f}")

# 3. 实际模型里的
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
tr = TinySD3Transformer2DModel.from_pretrained(
    f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr", subfolder="transformer",
    torch_dtype=torch.float16, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
for n,b in tr.named_buffers():
    if "pos_embed" in n:
        print(f"\n模型 buffer {n}  shape={tuple(b.shape)}  mean={b.float().mean():.6f} std={b.float().std():.6f}")
for n,pm in tr.named_parameters():
    if "pos_embed" in n:
        print(f"模型 param  {n}  shape={tuple(pm.shape)}  mean={pm.float().mean():.6f} std={pm.float().std():.6f}")

# 4. 判定：模型里的值是否等于缓存文件
import numpy as np
for n,b in list(tr.named_buffers())+list(tr.named_parameters()):
    if "pos_embed" in n and b.shape == pe.shape:
        d=(b.float().cpu()-pe.float()).abs().max().item()
        print(f"\n模型 {n} 与缓存文件最大差异 = {d:.8f}  -> {'一致(已正确加载)' if d<1e-5 else '不一致'}")
