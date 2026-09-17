"""排查无损提速测出 2.5-3.9% 数值差异的原因。

正常 fp16 jit 差异应在 0.1% 量级（之前提交模型实测 0.125%）。
疑点：baseline eager 39ms 远高于此前实测的 12-26ms，
      说明测试时 GPU 被占用严重，可能 ref 本身就是在异常状态下算的。
"""
import os, sys, torch, torch.nn as nn
import numpy as np
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0,TS); os.chdir(TS)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
dev,dt=torch.device("cuda"),torch.float16

occ=os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").read().strip()
print(f"GPU 占用: {occ or '无'}\n")

class M(nn.Module):
    def __init__(s,vae,tr,ts,pooled):
        super().__init__(); s.vae,s.tr=vae,tr
        s.register_buffer("ts",ts,persistent=False); s.register_buffer("pooled",pooled,persistent=False)
    def forward(s,x):
        sf=s.vae.config.scaling_factor
        z=s.vae.encode(x).latents*sf
        p=s.tr(hidden_states=z,timestep=s.ts,pooled_projections=s.pooled,return_dict=False)[0]
        return s.vae.decode((z-p)/sf,return_dict=False)[0].clamp(-1,1)

tr=TinySD3Transformer2DModel.from_pretrained(f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr",
    subfolder="transformer",torch_dtype=dt,low_cpu_mem_usage=False,ignore_mismatched_sizes=True)
vae=AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable",torch_dtype=dt)
lc=LoraConfig(r=64,lora_alpha=64,init_lora_weights="gaussian",
  target_modules=["to_k","to_q","to_v","to_out.0","proj","linear","linear_1","linear_2","net.2"])
tr.add_adapter(lc); tr.enable_adapters()
load_lora_state_dict(StableDiffusion3Pipeline.lora_state_dict(f"{TS}/checkpoint/tinysr",
    weight_name="transformer.safetensors"),tr)
vae=vae.to(dev,dtype=dt).eval(); tr=tr.to(dev,dtype=dt).eval()
ts=torch.tensor([1000.0],device=dev,dtype=dt)
pooled=torch.load(f"{TS}/dataset/default/pool_embeds.pt",map_location=dev).to(dtype=dt)
with torch.no_grad():
    w=torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
    zz=vae.encode(w).latents*vae.config.scaling_factor
    _=tr(hidden_states=zz,timestep=ts,pooled_projections=pooled,return_dict=False)
sd=torch.load(f"{ROOT}/train_out/tinysr10_bigdata/best.pth",map_location="cpu",weights_only=False)
cur=tr.state_dict()
for k,v in sd["lora"].items():
    if k in cur and cur[k].shape==v.shape: cur[k].copy_(v.to(cur[k].dtype))
tr=tr.merge_and_unload()
model=M(vae,tr,ts,pooled).to(dev).half().eval()

# 用真实图（随机噪声经 VAE 后行为异常，之前踩过）
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
a=np.asarray(Image.open(f"{ROOT}/data/赛题二/测试集/case1.jpg").convert("RGB"),dtype=np.float32)/255
x=((torch.from_numpy(a).permute(2,0,1)[None][:,:,1000:1512,1000:1512])*2-1).to(dev,dtype=dt)

print("=== 1. eager 自身的可复现性（同输入跑 5 次）===")
outs=[]
with torch.no_grad():
    for _ in range(5): outs.append(model(x).float().clone())
for i in range(1,5):
    d=(outs[i]-outs[0]).abs().mean().item()/outs[0].abs().mean().item()
    print(f"  第{i+1}次 vs 第1次: {d*100:.6f}%")
print("  -> eager 应完全确定（差异 0），若非零说明有非确定性算子")

print("\n=== 2. channels_last 差异（用真实图）===")
mcl=M(vae,tr,ts,pooled).to(dev).half().eval().to(memory_format=torch.channels_last)
with torch.no_grad():
    o=mcl(x.to(memory_format=torch.channels_last)).float()
d=(o-outs[0]).abs()
print(f"  相对 {d.mean().item()/outs[0].abs().mean().item()*100:.4f}%  最大 {d.max().item():.5f}")

print("\n=== 3. 数值差异对 Score 的实际影响 ===")
import pyiqa
lp=pyiqa.create_metric("lpips",device=dev)
gt=((torch.from_numpy(a).permute(2,0,1)[None][:,:,1000:1512,1000:1512])).to(dev)
o01=(outs[0]*0.5+0.5).clamp(0,1); c01=(o.clamp(-1,1)*0.5+0.5).clamp(0,1)
print(f"  LPIPS(eager, LQ)        {lp(o01,gt).item():.6f}")
print(f"  LPIPS(channels_last,LQ) {lp(c01,gt).item():.6f}")
print("  -> 若两者几乎相同，说明该量级的数值差异不影响指标")
