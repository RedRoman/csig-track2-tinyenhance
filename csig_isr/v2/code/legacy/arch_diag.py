"""架构诊断：定位改动应该落在哪里。

三个问题：
  1. DiT 各层的贡献度 —— 哪些层可以砍（层级重要性分析）
  2. latent 容量是否是瓶颈 —— 16 通道够不够
  3. VAE decoder 是否限制了上限 —— 用 GT 直接过 decoder 测天花板
"""
import os, sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0,TS); os.chdir(TS)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
dev,dt=torch.device("cuda"),torch.float16
VAL=f"{ROOT}/data/赛题二/验证集"

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
sd=torch.load(f"{ROOT}/train_out/run11_big_iqa20/best.pth",map_location="cpu",weights_only=False)
cur=tr.state_dict()
for k,v in sd["lora"].items():
    if k in cur and cur[k].shape==v.shape: cur[k].copy_(v.to(cur[k].dtype))
tr=tr.merge_and_unload()
sf=vae.config.scaling_factor
print(f"载入 run11 (step {sd.get('step')})，DiT {len(tr.transformer_blocks)} 层\n", flush=True)

def load_img(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]
PAIRS=[("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
       ("case3_lq.jpg","case3_gt.jpg",303)]
items=[]
for lqn,gtn,seed in PAIRS:
    lq_f,gt_f=load_img(f"{VAL}/{lqn}"),load_img(f"{VAL}/{gtn}")
    if lq_f.shape!=gt_f.shape: continue
    rng=np.random.RandomState(seed); _,_,H,W=gt_f.shape
    got,tries=0,0
    while got<6 and tries<90:
        tries+=1
        y,x=rng.randint(0,H-512),rng.randint(0,W-512)
        gt=gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item()<0.035: continue
        got+=1; items.append((lq_f[:,:,y:y+512,x:x+512].clone(), gt.clone()))
print(f"诊断样本 {len(items)} crop\n", flush=True)

import pyiqa
lp=pyiqa.create_metric("lpips",device=dev)
ds=pyiqa.create_metric("dists",device=dev)
def psnr(a,b): return -10*np.log10(F.mse_loss(a,b).item()+1e-12)

print("="*74)
print("诊断 1：VAE decoder 的重建天花板（GT 直接 encode->decode）")
print("="*74)
pl=[];ll=[]
with torch.no_grad():
    for lq,gt in items:
        g=gt.to(dev,dtype=dt)
        z=vae.encode(g*2-1).latents
        rec=(vae.decode(z,return_dict=False)[0].float()*0.5+0.5).clamp(0,1)
        pl.append(psnr(rec,gt.to(dev))); ll.append(lp(rec,gt.to(dev)).item())
print(f"  GT -> encode -> decode :  PSNR {np.mean(pl):.2f} dB   LPIPS {np.mean(ll):.4f}")
print(f"  => 这是【任何方案的上限】，因为输出必经此 decoder")

print()
print("="*74)
print("诊断 2：DiT 逐层贡献（跳过某层，看 LPIPS 恶化多少）")
print("="*74)
blocks=tr.transformer_blocks
n=len(blocks)
@torch.no_grad()
def run(lq01, skip=None):
    x=lq01.to(dev,dtype=dt)*2-1
    z=vae.encode(x).latents*sf
    h=tr.pos_embed(z)
    for i,b in enumerate(blocks):
        if skip is not None and i==skip: continue
        h=b.forward_(hidden_states=h)
    h=tr.norm(h)*(1+tr.scale)[:,None,:]+tr.shift[:,None,:]
    h=tr.proj_out(h)
    # unpatchify
    ps=tr.config.patch_size; hh=ww=64//ps
    h=h.reshape(1,hh,ww,ps,ps,16).permute(0,5,1,3,2,4).reshape(1,16,hh*ps,ww*ps)
    img=vae.decode((z-h)/sf,return_dict=False)[0]
    return (img.float()*0.5+0.5).clamp(0,1)

base=[]
with torch.no_grad():
    for lq,gt in items:
        base.append(lp(run(lq),gt.to(dev)).item())
b0=np.mean(base)
print(f"  完整 {n} 层 LPIPS = {b0:.4f}\n")
print(f"  {'跳过层':<10}{'LPIPS':>10}{'恶化':>10}{'重要性':>10}")
imp=[]
for i in range(n):
    vs=[]
    with torch.no_grad():
        for lq,gt in items:
            vs.append(lp(run(lq,skip=i),gt.to(dev)).item())
    d=np.mean(vs)-b0
    imp.append((i,d))
    bar="#"*max(int(d*300),0)
    print(f"  layer {i:<4}{np.mean(vs):>10.4f}{d:>+10.4f}  {bar}")
imp.sort(key=lambda t:t[1])
print(f"\n  最不重要的 4 层: {[t[0] for t in imp[:4]]}  (恶化 {[round(t[1],4) for t in imp[:4]]})")
print(f"  最重要的 3 层:   {[t[0] for t in imp[-3:]]}")
json.dump({"layer_importance":imp,"decoder_ceiling":{"psnr":float(np.mean(pl)),"lpips":float(np.mean(ll))}},
          open(f"{ROOT}/eval_out/arch_diag.json","w"),indent=1)
