"""信息瓶颈定位：分离 decoder 上限 / latent 容量 / DiT 预测能力 三者的贡献。

设计模块前必须知道：
  Q1. decoder 能表达的最好结果是什么？（给它最优 latent）
  Q2. DiT 预测的 latent 离最优 latent 有多远？
  Q3. 误差主要在哪个频段/哪类区域？
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
    while got<8 and tries<120:
        tries+=1
        y,x=rng.randint(0,H-512),rng.randint(0,W-512)
        gt=gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item()<0.035: continue
        got+=1; items.append((lq_f[:,:,y:y+512,x:x+512].clone(), gt.clone()))
print(f"样本 {len(items)} crop\n", flush=True)

import pyiqa
lpm=pyiqa.create_metric("lpips",device=dev)
def psnr(a,b): return -10*np.log10(F.mse_loss(a,b).item()+1e-12)

print("="*76)
print("Q1: decoder 的表达上限 —— 优化 latent 直接拟合 GT")
print("="*76)
res={}
pl=[];ll=[]
for lq,gt in items[:6]:
    g=gt.to(dev)
    z=torch.zeros(1,16,64,64,device=dev,dtype=torch.float32,requires_grad=True)
    with torch.no_grad():
        z.data=vae.encode(g.half()*2-1).latents.float()   # 用 GT 的 latent 作起点
    opt=torch.optim.Adam([z],lr=0.02)
    for _ in range(300):
        opt.zero_grad()
        out=(vae.decode(z.half(),return_dict=False)[0].float()*0.5+0.5).clamp(0,1)
        l=F.mse_loss(out,g)+0.5*lpm(out,g).mean()
        l.backward(); opt.step()
    with torch.no_grad():
        out=(vae.decode(z.half(),return_dict=False)[0].float()*0.5+0.5).clamp(0,1)
    pl.append(psnr(out,g)); ll.append(lpm(out,g).item())
res["decoder_ceiling"]={"psnr":float(np.mean(pl)),"lpips":float(np.mean(ll))}
print(f"  最优 latent -> decoder :  PSNR {np.mean(pl):.2f} dB   LPIPS {np.mean(ll):.4f}")
print(f"  => 这是 decoder 的硬上限，无论 DiT 多强都超不过\n", flush=True)

print("="*76)
print("Q2: 当前 DiT 输出 vs 最优 latent 的差距")
print("="*76)
pl2=[];ll2=[];zerr=[]
with torch.no_grad():
    for lq,gt in items[:6]:
        x=lq.to(dev,dtype=dt)*2-1; g=gt.to(dev)
        z_lq=vae.encode(x).latents*sf
        p=tr(hidden_states=z_lq,timestep=ts,pooled_projections=pooled,return_dict=False)[0]
        z_pred=z_lq-p
        z_gt=vae.encode(g.half()*2-1).latents*sf
        zerr.append((z_pred-z_gt).abs().mean().item()/z_gt.abs().mean().item())
        out=(vae.decode(z_pred/sf,return_dict=False)[0].float()*0.5+0.5).clamp(0,1)
        pl2.append(psnr(out,g)); ll2.append(lpm(out,g).item())
res["current"]={"psnr":float(np.mean(pl2)),"lpips":float(np.mean(ll2)),
                "latent_rel_err":float(np.mean(zerr))}
print(f"  当前模型输出       :  PSNR {np.mean(pl2):.2f} dB   LPIPS {np.mean(ll2):.4f}")
print(f"  预测 latent 相对误差:  {np.mean(zerr)*100:.2f}%")
gap_l=np.mean(ll2)-np.mean(ll)
print(f"\n  LPIPS 差距 {gap_l:+.4f}  => {'DiT 预测是瓶颈' if gap_l>0.05 else 'decoder 是瓶颈'}\n", flush=True)

print("="*76)
print("Q3: 误差的频段分布（当前输出 vs GT）")
print("="*76)
def bands(x,k=(4,16)):
    g=x.mean(1,keepdim=True)
    lo=F.interpolate(F.avg_pool2d(g,k[1]),size=g.shape[-2:],mode="bilinear",align_corners=False)
    mid=F.interpolate(F.avg_pool2d(g,k[0]),size=g.shape[-2:],mode="bilinear",align_corners=False)-lo
    hi=g-lo-mid
    return lo,mid,hi
el=[];em=[];eh=[]
with torch.no_grad():
    for lq,gt in items[:6]:
        x=lq.to(dev,dtype=dt)*2-1; g=gt.to(dev)
        z=vae.encode(x).latents*sf
        p=tr(hidden_states=z,timestep=ts,pooled_projections=pooled,return_dict=False)[0]
        out=(vae.decode((z-p)/sf,return_dict=False)[0].float()*0.5+0.5).clamp(0,1)
        ol,om,oh=bands(out); gl,gm,gh=bands(g)
        el.append((ol-gl).abs().mean().item()); em.append((om-gm).abs().mean().item())
        eh.append((oh-gh).abs().mean().item())
tot=np.mean(el)+np.mean(em)+np.mean(eh)
print(f"  低频误差 {np.mean(el):.5f}  ({np.mean(el)/tot*100:.1f}%)")
print(f"  中频误差 {np.mean(em):.5f}  ({np.mean(em)/tot*100:.1f}%)")
print(f"  高频误差 {np.mean(eh):.5f}  ({np.mean(eh)/tot*100:.1f}%)")
res["band_err"]={"low":float(np.mean(el)),"mid":float(np.mean(em)),"high":float(np.mean(eh))}
json.dump(res,open(f"{ROOT}/eval_out/bottleneck.json","w"),indent=1)
