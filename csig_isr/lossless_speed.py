"""无损提速尝试：不改模型结构与权重，只做编译/内存布局优化。

背景：反推公式 榜分=(Score/6*100)*加速比^0.0708，
      任何伤及 Score 的提速都不划算（实测快速 encoder 净亏 0.86 分）。
      但【纯编译优化】理论上数值不变，是白赚的。

测试项：
  1. baseline eager
  2. channels_last 内存布局
  3. torch.jit.trace + freeze（当前提交用的）
  4. torch.compile (inductor)
  5. SDPA flash backend
每项都验证与 baseline 的数值差异。
"""
import os, sys, time, json
import numpy as np, torch, torch.nn as nn
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0,TS); os.chdir(TS)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
dev,dt=torch.device("cuda"),torch.float16

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

x=torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
with torch.no_grad(): ref=model(x).float().clone()

def bench(f,inp,n=100,warm=30):
    with torch.no_grad():
        for _ in range(warm): f(inp)
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): f(inp)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n
def stable(f,inp,rounds=5): return min(bench(f,inp) for _ in range(rounds))

print(f"{'方案':<32}{'ms':>9}{'加速':>8}{'数值差异':>12}{'榜分变化':>10}")
print("-"*72)
S=3.7649
def board(sp): return (S/6*100)*sp**0.0708*0.9937
t0=stable(model,x); b0=board(99.96/t0)
print(f"{'baseline eager':<32}{t0:>9.3f}{'1.00x':>8}{'--':>12}{'--':>10}")

results={"eager":t0}
# channels_last
try:
    mcl=M(vae,tr,ts,pooled).to(dev).half().eval().to(memory_format=torch.channels_last)
    xcl=x.to(memory_format=torch.channels_last)
    with torch.no_grad(): o=mcl(xcl).float()
    d=(o-ref).abs().mean().item()/ref.abs().mean().item()
    t=stable(mcl,xcl); results["channels_last"]=t
    print(f"{'channels_last':<32}{t:>9.3f}{t0/t:>7.2f}x{d*100:>11.4f}%{board(99.96/t)-b0:>+10.2f}")
except Exception as e:
    print(f"{'channels_last':<32}  失败 {str(e)[:40]}")
# jit
try:
    with torch.no_grad():
        j=torch.jit.freeze(torch.jit.trace(model,x,check_trace=False).eval(),optimize_numerics=False)
        for _ in range(30): j(x)
        o=j(x).float()
    d=(o-ref).abs().mean().item()/ref.abs().mean().item()
    t=stable(j,x); results["jit"]=t
    print(f"{'jit trace+freeze':<32}{t:>9.3f}{t0/t:>7.2f}x{d*100:>11.4f}%{board(99.96/t)-b0:>+10.2f}")
except Exception as e:
    print(f"{'jit':<32}  失败 {str(e)[:40]}")
# torch.compile
for mode in ["default","max-autotune-no-cudagraphs"]:
    try:
        torch._dynamo.reset()
        mc=torch.compile(M(vae,tr,ts,pooled).to(dev).half().eval(),mode=mode,fullgraph=False)
        with torch.no_grad():
            for _ in range(5): mc(x)
            o=mc(x).float()
        d=(o-ref).abs().mean().item()/ref.abs().mean().item()
        t=stable(mc,x); results[f"compile_{mode}"]=t
        print(f"{'torch.compile '+mode:<32}{t:>9.3f}{t0/t:>7.2f}x{d*100:>11.4f}%{board(99.96/t)-b0:>+10.2f}")
    except Exception as e:
        print(f"{'torch.compile '+mode:<32}  失败 {str(e)[:40]}")
json.dump(results,open(f"{ROOT}/eval_out/lossless_speed.json","w"),indent=1)
