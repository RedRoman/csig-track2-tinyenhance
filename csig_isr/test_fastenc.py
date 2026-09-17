"""端到端验证：用小网络替代 VAE encoder 后，Score 掉多少。

latent 相对误差 15.6% 看似不小，但 DiT 做的是残差预测（z - pred），
误差可能被部分抵消，也可能被放大。必须实测 Score，不能只看拟合 R^2。
"""
import os, sys, glob, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0,TS); os.chdir(TS)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from utils.wavelet_color_fix import wavelet_color_fix
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
from torchvision.transforms import ToPILImage, ToTensor
dev,dt=torch.device("cuda"),torch.float16
VAL=f"{ROOT}/data/赛题二/验证集"

# ---- 载入完整模型 ----
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
sd=torch.load(f"{ROOT}/train_out/tinysr8_fid/best.pth",map_location="cpu",weights_only=False)
cur=tr.state_dict()
for k,v in sd["lora"].items():
    if k in cur and cur[k].shape==v.shape: cur[k].copy_(v.to(cur[k].dtype))

# ---- 训练替代 encoder（2层 c64，更多步数）----
fast=nn.Sequential(nn.PixelUnshuffle(8),
                   nn.Conv2d(192,64,3,padding=1), nn.SiLU(),
                   nn.Conv2d(64,64,3,padding=1), nn.SiLU(),
                   nn.Conv2d(64,16,1)).to(dev).float()
fs=sorted(glob.glob(f"{ROOT}/AdcSR/testset/DIV2K-Val/HR/*.png"))[:800]
imgs=[]
for f in fs:
    a=np.asarray(Image.open(f).convert("RGB"),dtype=np.float32)/255.
    if a.shape[0]>=512 and a.shape[1]>=512: imgs.append(torch.from_numpy(a[:512,:512]).permute(2,0,1))
X=torch.stack(imgs); print(f"蒸馏样本 {X.shape[0]} 张", flush=True)
opt=torch.optim.AdamW(fast.parameters(),lr=2e-3)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=3000)
for step in range(3000):
    idx=torch.randint(0,X.shape[0],(8,)); xb=X[idx].to(dev)*2-1
    with torch.no_grad(): zt=vae.encode(xb.half()).latents.float()
    loss=F.mse_loss(fast(xb),zt)
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if (step+1)%1000==0: print(f"  step {step+1} loss {loss.item():.6f}", flush=True)
fast=fast.half().eval()
with torch.no_grad():
    xb=X[:32].to(dev)*2-1
    zt=vae.encode(xb.half()).latents.float(); zp=fast(xb.half()).float()
    r2=1-((zp-zt)**2).sum()/((zt-zt.mean())**2).sum()
print(f"蒸馏后 R^2 = {r2.item():.4f}", flush=True)

sf=vae.config.scaling_factor
@torch.no_grad()
def infer(lq01, use_fast):
    x=lq01.to(dev,dtype=dt)*2-1
    z=(fast(x) if use_fast else vae.encode(x).latents)*sf
    p=tr(hidden_states=z,timestep=ts,pooled_projections=pooled,return_dict=False)[0]
    img=vae.decode((z-p)/sf,return_dict=False)[0]
    sr=(img.float()*0.5+0.5).clamp(0,1)
    f=wavelet_color_fix(ToPILImage()(sr[0].cpu()), ToPILImage()(lq01[0].cpu()))
    return ToTensor()(f)[None].to(dev)

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
print(f"验证 {len(items)} crop", flush=True)

import pyiqa
M={"lpips":pyiqa.create_metric("lpips",device=dev),"dists":pyiqa.create_metric("dists",device=dev),
   "niqe":pyiqa.create_metric("niqe",device=dev),"musiq":pyiqa.create_metric("musiq",device=dev),
   "maniqa":pyiqa.create_metric("maniqa-pipal",device=dev),"clipiqa":pyiqa.create_metric("clipiqa",device=dev),
   "psnr":pyiqa.create_metric("psnr",device=dev,test_y_channel=True,color_space="ycbcr")}
def ev(use_fast,tag):
    res={k:[] for k in M}
    for lq,gt in items:
        o=infer(lq,use_fast); g=gt.to(dev)
        for k,f in M.items():
            res[k].append((f(o,g) if k in ("lpips","dists","psnr") else f(o)).item())
        torch.cuda.empty_cache()
    m={k:float(np.mean(v)) for k,v in res.items()}
    m["Score"]=((1-m["lpips"])+(1-m["dists"])+m["clipiqa"]+m["maniqa"]+m["musiq"]/100
                +max(0,(10-m["niqe"])/10))
    print(f"  {tag:<22}Score {m['Score']:.4f}  PSNR {m['psnr']:.2f}  "
          f"lpips {m['lpips']:.4f} dists {m['dists']:.4f} clipiqa {m['clipiqa']:.4f}", flush=True)
    return m
print()
a=ev(False,"原版 VAE encoder")
b=ev(True, "快速 encoder")
print(f"\n  Score 变化 {b['Score']-a['Score']:+.4f}")

def bench(f,n=60,warm=20):
    x=torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
    with torch.no_grad():
        for _ in range(warm): f(x)
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): f(x)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n
t1=min(bench(lambda a: vae.encode(a)) for _ in range(3))
t2=min(bench(lambda a: fast(a)) for _ in range(3))
print(f"  encoder 时延 {t1:.3f} -> {t2:.3f} ms  省 {t1-t2:.2f} ms")
tot_old=12.20; tot_new=tot_old-(t1-t2)
sp_old, sp_new = 99.96/tot_old, 99.96/tot_new
print(f"  总时延 {tot_old:.2f} -> {tot_new:.2f} ms   加速比 {sp_old:.2f}x -> {sp_new:.2f}x")
bo=(a['Score']/6*100)*sp_old**0.0708*0.9937
bn=(b['Score']/6*100)*sp_new**0.0708*0.9937
print(f"  预估榜分 {bo:.2f} -> {bn:.2f}  ({bn-bo:+.2f})")
torch.save(fast.state_dict(), f"{ROOT}/x1_model/fast_encoder.pth")
