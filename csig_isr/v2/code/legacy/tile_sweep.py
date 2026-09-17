"""4K 出图的 tile 策略扫描 —— 这部分【不计入时延】，是纯免费的质量提升。

赛题原文：「Diffusion 的输入和输出分辨率均为 512x512，在单块显卡上评测时延」
=> 时延只测单个 512 tile 的前向，4K 图怎么切、怎么融合完全不影响加速比。

扫描维度：
  - overlap: 64 / 128 / 192 / 256
  - 融合权重: 高斯 / Hann
  - 是否镜像 padding
用 3 组真实配对（有 GT）算 Score 选最优。
"""
import os, sys, json, itertools
import numpy as np, torch, torch.nn.functional as F
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
CKPT=sys.argv[1] if len(sys.argv)>1 else f"{ROOT}/train_out/tinysr8_fid/best.pth"

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
sd=torch.load(CKPT,map_location="cpu",weights_only=False)
cur=tr.state_dict()
for k,v in sd["lora"].items():
    if k in cur and cur[k].shape==v.shape: cur[k].copy_(v.to(cur[k].dtype))
tr=tr.merge_and_unload()
print(f"载入 {os.path.basename(os.path.dirname(CKPT))} step {sd.get('step')}", flush=True)
sf=vae.config.scaling_factor

@torch.no_grad()
def core(x01):
    x=x01*2-1
    z=vae.encode(x).latents*sf
    p=tr(hidden_states=z,timestep=ts,pooled_projections=pooled,return_dict=False)[0]
    return (vae.decode((z-p)/sf,return_dict=False)[0].float()*0.5+0.5).clamp(0,1)

def w_gauss(n, ov, sigma=0.45):
    t=torch.linspace(-1,1,n,device=dev)
    return torch.exp(-(t**2)/(2*sigma**2))
def w_hann(n, ov):
    v=torch.ones(n,device=dev)
    if ov>0:
        t=torch.linspace(0,np.pi,2*ov,device=dev)
        e=(1-torch.cos(t))/2
        v[:ov]=e[:ov]; v[-ov:]=e[ov:].flip(0)
    return v

@torch.no_grad()
def tiled(img01, tile, ov, wfn, pad_mode):
    _,_,H0,W0=img01.shape
    pad = ov//2 if pad_mode else 0
    if pad: img01=F.pad(img01,(pad,pad,pad,pad),mode="reflect")
    _,_,H,W=img01.shape
    st=tile-ov
    ys=list(range(0,max(H-tile,0)+1,st)); xs=list(range(0,max(W-tile,0)+1,st))
    if not ys or ys[-1]+tile<H: ys.append(max(H-tile,0))
    if not xs or xs[-1]+tile<W: xs.append(max(W-tile,0))
    wy,wx=wfn(tile,ov),wfn(tile,ov)
    wm=(wy[:,None]*wx[None,:])[None,None].float()
    out=torch.zeros(1,3,H,W,device=dev,dtype=torch.float32)
    acc=torch.zeros(1,1,H,W,device=dev,dtype=torch.float32)
    for y in ys:
        for x in xs:
            o=core(img01[:,:,y:y+tile,x:x+tile].to(dev,dtype=dt))
            out[:,:,y:y+tile,x:x+tile]+=o*wm
            acc[:,:,y:y+tile,x:x+tile]+=wm
    r=(out/acc.clamp(min=1e-8)).clamp(0,1)
    return r[:,:,pad:pad+H0,pad:pad+W0] if pad else r

def load_img(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]

PAIRS=[("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]
# 用 1024 区域测（全 4K 太慢，1024 足够体现 tile 融合差异）
pairs=[]
for lqn,gtn in PAIRS:
    lq=load_img(f"{VAL}/{lqn}"); gt=load_img(f"{VAL}/{gtn}")
    if lq.shape!=gt.shape: continue
    _,_,H,W=gt.shape
    y,x=H//2-512,W//2-512
    pairs.append((lq[:,:,y:y+1024,x:x+1024].clone(), gt[:,:,y:y+1024,x:x+1024].clone()))
print(f"测试区域 {len(pairs)} 个 1024x1024", flush=True)

import pyiqa
M={"lpips":pyiqa.create_metric("lpips",device=dev),"dists":pyiqa.create_metric("dists",device=dev),
   "niqe":pyiqa.create_metric("niqe",device=dev),"musiq":pyiqa.create_metric("musiq",device=dev),
   "maniqa":pyiqa.create_metric("maniqa-pipal",device=dev),"clipiqa":pyiqa.create_metric("clipiqa",device=dev),
   "psnr":pyiqa.create_metric("psnr",device=dev,test_y_channel=True,color_space="ycbcr")}

print(f"\n{'配置':<30}{'Score':>9}{'PSNR':>8}{'tile数':>8}")
print("-"*57)
best=None
for ov, wname, padm in itertools.product([64,128,192,256],["gauss","hann"],[False,True]):
    wfn = w_gauss if wname=="gauss" else w_hann
    res={k:[] for k in M}
    ntile=0
    for lq,gt in pairs:
        o=tiled(lq,512,ov,wfn,padm)
        f=wavelet_color_fix(ToPILImage()(o[0].cpu()), ToPILImage()(lq[0].cpu()))
        o=ToTensor()(f)[None].to(dev)
        g=gt.to(dev)
        for k,fn in M.items():
            res[k].append((fn(o,g) if k in ("lpips","dists","psnr") else fn(o)).item())
        st=512-ov
        ntile=((1024-512)//st+2)**2
        torch.cuda.empty_cache()
    m={k:float(np.mean(v)) for k,v in res.items()}
    sc=((1-m["lpips"])+(1-m["dists"])+m["clipiqa"]+m["maniqa"]+m["musiq"]/100
        +max(0,(10-m["niqe"])/10))
    tag=f"ov{ov} {wname}{' +pad' if padm else ''}"
    print(f"{tag:<30}{sc:>9.4f}{m['psnr']:>8.2f}{ntile:>8}", flush=True)
    if best is None or sc>best[0]: best=(sc,tag,ov,wname,padm)
print(f"\n最优: {best[1]}  Score {best[0]:.4f}")
json.dump({"best":{"score":best[0],"tag":best[1],"overlap":best[2],
                   "weight":best[3],"pad":best[4]}},
          open(f"{ROOT}/eval_out/tile_sweep.json","w"),indent=1)
