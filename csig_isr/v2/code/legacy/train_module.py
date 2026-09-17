"""训练新模块（HF旁路 / 频域融合），验证能否覆盖时延成本。

设计依据（来自瓶颈诊断）：
  - decoder 是硬瓶颈（上限 LPIPS 0.5167，当前 0.5003 已反超）
  - 误差 81% 在中高频，低频仅 19%
  - decoder 的 8x 上采样天然低通 => 高频应走独立的像素域路径

准入门槛（时延每 +1ms 扣 0.415 分）：
  HF旁路 ch16x3 (1.66ms) 需 Score +0.045
  频域融合 ch16 (0.93ms) 需 Score +0.025
"""
import os, sys, io, glob, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
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

ap=argparse.ArgumentParser()
ap.add_argument("--module",choices=["hf","freq","both"],default="hf")
ap.add_argument("--steps",type=int,default=1500)
ap.add_argument("--bs",type=int,default=4)
ap.add_argument("--lr",type=float,default=2e-4)
ap.add_argument("--w_iqa",type=float,default=3.0)
ap.add_argument("--out",type=str,default=f"{ROOT}/train_out/mod_hf")
args=ap.parse_args()
os.makedirs(args.out,exist_ok=True)

class HFBypass(nn.Module):
    def __init__(s,ch=16,n=3):
        super().__init__()
        s.head=nn.Conv2d(3,ch,3,padding=1)
        s.body=nn.ModuleList([nn.Sequential(nn.Conv2d(ch,ch,3,padding=1),nn.SiLU(),
                                            nn.Conv2d(ch,ch,3,padding=1)) for _ in range(n)])
        s.tail=nn.Conv2d(ch,3,3,padding=1)
        nn.init.zeros_(s.tail.weight); nn.init.zeros_(s.tail.bias)
        s.act=nn.SiLU()
    def forward(s,lq,dec):
        f=s.act(s.head(lq.float()))
        for b in s.body: f=f+b(f)
        return dec+s.tail(f)

class FreqFusion(nn.Module):
    def __init__(s,ch=16):
        super().__init__()
        s.g=nn.Sequential(nn.Conv2d(6,ch,3,padding=1),nn.SiLU(),
                          nn.Conv2d(ch,ch,3,padding=1),nn.SiLU(),
                          nn.Conv2d(ch,3,3,padding=1),nn.Sigmoid())
        nn.init.zeros_(s.g[-2].weight); nn.init.constant_(s.g[-2].bias,4.0)
    def forward(s,lq,dec,k=16):
        lq=lq.float(); dec=dec.float()
        lo_l=F.interpolate(F.avg_pool2d(lq,k),size=lq.shape[-2:],mode="bilinear",align_corners=False)
        lo_d=F.interpolate(F.avg_pool2d(dec,k),size=lq.shape[-2:],mode="bilinear",align_corners=False)
        w=s.g(torch.cat([lq,dec],1))
        return (w*lo_d+(1-w)*lo_l)+(dec-lo_d)

def degrade(img,rng):
    x=img
    if rng.random()<0.9:
        s=float(rng.uniform(40,100)); x=cv2.bilateralFilter(x,9,s,s)
    x=cv2.GaussianBlur(x.astype(np.float32),(0,0),float(rng.uniform(0.8,1.6)))
    x=np.clip(x,0,255).astype(np.uint8)
    buf=io.BytesIO(); Image.fromarray(x).save(buf,format="JPEG",quality=int(rng.uniform(50,95)))
    buf.seek(0); return np.asarray(Image.open(buf).convert("RGB"),dtype=np.uint8)

class DS(Dataset):
    def __init__(s,paths): s.p=paths
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        rng=np.random.RandomState((int(torch.initial_seed())%99991*7919+i*104729)%(2**31))
        gt=np.asarray(Image.open(s.p[i]).convert("RGB"),dtype=np.uint8)
        h,w,_=gt.shape
        if h>512 or w>512:
            y,x=rng.randint(0,max(h-512,0)+1),rng.randint(0,max(w-512,0)+1)
            gt=gt[y:y+512,x:x+512]
        if gt.shape[0]!=512 or gt.shape[1]!=512:
            gt=cv2.resize(gt,(512,512),interpolation=cv2.INTER_CUBIC)
        if rng.random()<0.5: gt=gt[:,::-1]
        gt=np.ascontiguousarray(gt); lq=degrade(gt,rng)
        t=lambda a: torch.from_numpy(a.astype(np.float32)/255.).permute(2,0,1)
        return t(lq),t(gt)

# 载入冻结的主干
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
    w0=torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
    zz=vae.encode(w0).latents*vae.config.scaling_factor
    _=tr(hidden_states=zz,timestep=ts,pooled_projections=pooled,return_dict=False)
sd=torch.load(f"{ROOT}/train_out/run12_iqa30/best.pth",map_location="cpu",weights_only=False)
cur=tr.state_dict()
for k,v in sd["lora"].items():
    if k in cur and cur[k].shape==v.shape: cur[k].copy_(v.to(cur[k].dtype))
tr=tr.merge_and_unload()
for p in list(tr.parameters())+list(vae.parameters()): p.requires_grad_(False)
sf=vae.config.scaling_factor
print(f"主干载自 run12 (step {sd.get('step')})，已冻结",flush=True)

mods=nn.ModuleDict()
if args.module in ("hf","both"): mods["hf"]=HFBypass(16,3).to(dev).float()
if args.module in ("freq","both"): mods["freq"]=FreqFusion(16).to(dev).float()
npar=sum(p.numel() for p in mods.parameters())
print(f"新模块 {list(mods.keys())}  参数 {npar/1e3:.1f}K",flush=True)

@torch.no_grad()
def backbone(lq01):
    x=lq01.to(dev,dtype=dt)*2-1
    z=vae.encode(x).latents*sf
    p=tr(hidden_states=z,timestep=ts,pooled_projections=pooled,return_dict=False)[0]
    return vae.decode((z-p)/sf,return_dict=False)[0].float()

def full(lq01):
    d=backbone(lq01)
    if "freq" in mods: d=mods["freq"](lq01.to(dev)*2-1,d)
    if "hf" in mods:   d=mods["hf"](lq01.to(dev)*2-1,d)
    return (d*0.5+0.5).clamp(0,1)

def load_img(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]
PAIRS=[("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
       ("case3_lq.jpg","case3_gt.jpg",303)]
val=[]
for lqn,gtn,seed in PAIRS:
    lf,gf=load_img(f"{VAL}/{lqn}"),load_img(f"{VAL}/{gtn}")
    if lf.shape!=gf.shape: continue
    rng=np.random.RandomState(seed); _,_,H,W=gf.shape
    got,tries=0,0
    while got<10 and tries<150:
        tries+=1
        y,x=rng.randint(0,H-512),rng.randint(0,W-512)
        g=gf[:,:,y:y+512,x:x+512]
        if g.std().item()<0.035: continue
        got+=1; val.append((lf[:,:,y:y+512,x:x+512].clone(),g.clone()))
print(f"验证 {len(val)} crop",flush=True)

import pyiqa
MV={"lpips":pyiqa.create_metric("lpips",device=dev),"dists":pyiqa.create_metric("dists",device=dev),
    "niqe":pyiqa.create_metric("niqe",device=dev),"musiq":pyiqa.create_metric("musiq",device=dev),
    "maniqa":pyiqa.create_metric("maniqa-pipal",device=dev),"clipiqa":pyiqa.create_metric("clipiqa",device=dev)}
@torch.no_grad()
def evaluate():
    mods.eval(); r={k:[] for k in MV}
    for lq,gt in val:
        o=full(lq); g=gt.to(dev)
        for k,f in MV.items():
            r[k].append((f(o,g) if k in ("lpips","dists") else f(o)).item())
        torch.cuda.empty_cache()
    mods.train()
    m={k:float(np.mean(v)) for k,v in r.items()}
    return ((1-m["lpips"])+(1-m["dists"])+m["clipiqa"]+m["maniqa"]+m["musiq"]/100
            +max(0,(10-m["niqe"])/10))

lp_l=pyiqa.create_metric("lpips",device=dev,as_loss=True)
ds_l=pyiqa.create_metric("dists",device=dev,as_loss=True)
ci_l=pyiqa.create_metric("clipiqa",device=dev,as_loss=True)
mu_l=pyiqa.create_metric("musiq",device=dev,as_loss=True)

paths=(sorted(glob.glob(f"{ROOT}/AdcSR/testset/DIV2K-Val/HR/*.png"))
       +sorted(glob.glob(f"{ROOT}/train_data/lsdir512/*.png")))
print(f"训练图 {len(paths)}",flush=True)
dl=DataLoader(DS(paths),batch_size=args.bs,shuffle=True,num_workers=6,drop_last=True,
              pin_memory=True,persistent_workers=True)
opt=torch.optim.AdamW(mods.parameters(),lr=args.lr,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.steps,eta_min=args.lr*0.05)

s0=evaluate(); print(f"[step 0] Score {s0:.4f}",flush=True)
best=s0; step=0; t0=time.time()
while step<args.steps:
    for lq,gt in dl:
        if step>=args.steps: break
        lq,gt=lq.to(dev,non_blocking=True),gt.to(dev,non_blocking=True)
        out=full(lq)
        l1=F.l1_loss(out,gt); lp=lp_l(out,gt).mean(); dsv=ds_l(out,gt).mean()
        loss=0.3*l1+1.0*lp+1.5*dsv-args.w_iqa*(ci_l(out).mean()+mu_l(out).mean()/100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(mods.parameters(),1.0); opt.step(); sch.step()
        step+=1
        if step%50==0:
            print(f"[{step}/{args.steps}] loss {loss.item():.4f} {(time.time()-t0)/step:.2f}s/it",flush=True)
        if step%150==0:
            sc=evaluate(); flag=""
            if sc>best:
                best=sc; torch.save(mods.state_dict(),f"{args.out}/best.pth"); flag="  <- best"
            print(f"[val {step}] Score {sc:.4f}{flag}",flush=True)
print(f"完成 best {best:.4f}  (起点 {s0:.4f}, 增益 {best-s0:+.4f})",flush=True)
