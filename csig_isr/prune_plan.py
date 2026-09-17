"""基于层重要性的剪枝方案验证 + 修正 VAE 往返测试。

诊断 2 结论：layer 3/5 为负贡献（跳过反而 LPIPS 更好），
layer 1/6/2/4/11 贡献 <0.01，layer 0/7/8/9 是骨干。

本脚本：
  1. 修正 VAE 往返重建测试（之前 scaling_factor 用错）
  2. 逐个验证剪枝组合的 Score 与时延
"""
import os, sys, json, itertools
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
blocks=tr.transformer_blocks

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
    while got<10 and tries<150:
        tries+=1
        y,x=rng.randint(0,H-512),rng.randint(0,W-512)
        gt=gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item()<0.035: continue
        got+=1; items.append((lq_f[:,:,y:y+512,x:x+512].clone(), gt.clone()))

print("="*70)
print("修正诊断 1：VAE 往返重建（正确使用 scaling_factor）")
print("="*70)
pl=[];ll=[]
import pyiqa
lpm=pyiqa.create_metric("lpips",device=dev)
with torch.no_grad():
    for lq,gt in items[:8]:
        g=gt.to(dev,dtype=dt)
        z=vae.encode(g*2-1).latents          # AutoencoderTiny 的 scaling_factor=1.0
        rec=(vae.decode(z,return_dict=False)[0].float()*0.5+0.5).clamp(0,1)
        gg=gt.to(dev)
        pl.append(-10*np.log10(F.mse_loss(rec,gg).item()+1e-12))
        ll.append(lpm(rec,gg).item())
print(f"  GT->encode->decode:  PSNR {np.mean(pl):.2f} dB   LPIPS {np.mean(ll):.4f}")
print(f"  （scaling_factor={sf}，AutoencoderTiny 为 1.0 故无需缩放）")

print()
print("="*70)
print("剪枝方案验证（跳过指定层，不重训）")
print("="*70)
@torch.no_grad()
def run(lq01, skip=()):
    x=lq01.to(dev,dtype=dt)*2-1
    z=vae.encode(x).latents*sf
    h=tr.pos_embed(z)
    for i,b in enumerate(blocks):
        if i in skip: continue
        h=b.forward_(hidden_states=h)
    h=tr.norm(h)*(1+tr.scale)[:,None,:]+tr.shift[:,None,:]
    h=tr.proj_out(h)
    ps=tr.config.patch_size; hh=ww=64//ps
    h=h.reshape(1,hh,ww,ps,ps,16).permute(0,5,1,3,2,4).reshape(1,16,hh*ps,ww*ps)
    img=vae.decode((z-h)/sf,return_dict=False)[0]
    sr=(img.float()*0.5+0.5).clamp(0,1)
    f=wavelet_color_fix(ToPILImage()(sr[0].cpu()),ToPILImage()(lq01[0].cpu()))
    return ToTensor()(f)[None].to(dev)

M={"lpips":lpm,"dists":pyiqa.create_metric("dists",device=dev),
   "niqe":pyiqa.create_metric("niqe",device=dev),"musiq":pyiqa.create_metric("musiq",device=dev),
   "maniqa":pyiqa.create_metric("maniqa-pipal",device=dev),"clipiqa":pyiqa.create_metric("clipiqa",device=dev)}
def ev(skip):
    r={k:[] for k in M}
    for lq,gt in items:
        o=run(lq,skip); g=gt.to(dev)
        for k,f in M.items():
            r[k].append((f(o,g) if k in ("lpips","dists") else f(o)).item())
        torch.cuda.empty_cache()
    m={k:float(np.mean(v)) for k,v in r.items()}
    return ((1-m["lpips"])+(1-m["dists"])+m["clipiqa"]+m["maniqa"]+m["musiq"]/100
            +max(0,(10-m["niqe"])/10)), m

def bench(skip,n=50,warm=20):
    x=torch.rand(1,3,512,512,device=dev,dtype=dt)
    with torch.no_grad():
        for _ in range(warm): run_raw(x,skip)
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): run_raw(x,skip)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n
@torch.no_grad()
def run_raw(x01, skip):
    x=x01*2-1
    z=vae.encode(x).latents*sf
    h=tr.pos_embed(z)
    for i,b in enumerate(blocks):
        if i in skip: continue
        h=b.forward_(hidden_states=h)
    h=tr.norm(h)*(1+tr.scale)[:,None,:]+tr.shift[:,None,:]
    h=tr.proj_out(h)
    ps=tr.config.patch_size; hh=ww=64//ps
    h=h.reshape(1,hh,ww,ps,ps,16).permute(0,5,1,3,2,4).reshape(1,16,hh*ps,ww*ps)
    return vae.decode((z-h)/sf,return_dict=False)[0]

plans=[((),"完整 12 层"), ((5,),"去 L5"), ((3,5),"去 L3,5"), ((1,3,5),"去 L1,3,5"),
       ((1,3,5,6),"去 L1,3,5,6"), ((1,2,3,4,5,6),"去 L1-6 弱层")]
print(f"  {'方案':<16}{'层数':>6}{'Score':>9}{'时延ms':>9}{'加速比':>8}{'预估榜分':>10}")
print("  "+"-"*58)
for skip,tag in plans:
    sc,m=ev(skip)
    t=min(bench(skip) for _ in range(3))
    spd=99.96/t
    bd=(sc/6*100)*spd**0.0708*0.9937
    print(f"  {tag:<16}{12-len(skip):>6}{sc:>9.4f}{t:>9.2f}{spd:>7.2f}x{bd:>10.2f}", flush=True)
