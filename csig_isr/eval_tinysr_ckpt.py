"""评测 TinySR 训练后的 checkpoint（Score 口径 + 色彩对齐变体）。"""
import os, sys, json, argparse
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
TS = f"{ROOT}/TinySR"
sys.path.insert(0, TS); os.chdir(TS)

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
from torchvision.transforms import ToPILImage, ToTensor

VAL = f"{ROOT}/data/赛题二/验证集"
dev, dt = torch.device("cuda"), torch.float16
PAIRS = [("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
         ("case3_lq.jpg","case3_gt.jpg",303)]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, default=f"{ROOT}/train_out/tinysr1/best.pth")
ap.add_argument("--n", type=int, default=6)
args = ap.parse_args()


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1)[None]


tr = TinySD3Transformer2DModel.from_pretrained(
    f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr", subfolder="transformer",
    torch_dtype=dt, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
vae = AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable", torch_dtype=dt)
lc = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian",
                target_modules=["to_k","to_q","to_v","to_out.0","proj","linear","linear_1","linear_2","net.2"])
tr.add_adapter(lc); tr.enable_adapters()
load_lora_state_dict(StableDiffusion3Pipeline.lora_state_dict(
    f"{TS}/checkpoint/tinysr", weight_name="transformer.safetensors"), tr)
vae = vae.to(dev,dtype=dt).eval(); tr = tr.to(dev,dtype=dt).eval()
ts = torch.tensor([1000.0], device=dev, dtype=dt)
pooled = torch.load(f"{TS}/dataset/default/pool_embeds.pt", map_location=dev).to(dtype=dt)

# warmup 触发缓存（必须在载入训练后 LoRA 之前完成结构固化）
with torch.no_grad():
    w = torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
    zz = vae.encode(w).latents*vae.config.scaling_factor
    _ = tr(hidden_states=zz, timestep=ts, pooled_projections=pooled, return_dict=False)

# 载入训练后的 LoRA
if args.ckpt and os.path.exists(args.ckpt):
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    lora = sd["lora"]
    cur = tr.state_dict()
    hit = 0
    for k, v in lora.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k].copy_(v.to(cur[k].dtype)); hit += 1
    print(f"载入 {args.ckpt} (step {sd.get('step')})  匹配 {hit}/{len(lora)} 个 LoRA 张量")
else:
    print("未载入训练权重（zero-shot）")


@torch.no_grad()
def raw(lq01):
    x = lq01.to(dev,dtype=dt)*2-1
    z = vae.encode(x).latents*vae.config.scaling_factor
    p = tr(hidden_states=z, timestep=ts, pooled_projections=pooled, return_dict=False)[0]
    img = vae.decode((z-p)/vae.config.scaling_factor, return_dict=False)[0]
    return (img.float()*0.5+0.5).clamp(0,1)


def fixed(lq01, method):
    sr = raw(lq01)
    f = (adain_color_fix if method=="adain" else wavelet_color_fix)(
        ToPILImage()(sr[0].cpu()), ToPILImage()(lq01[0].cpu()))
    return ToTensor()(f)[None].to(dev)


crops=[]
for lqn, gtn, seed in PAIRS:
    lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
    if lq_f.shape != gt_f.shape: continue
    rng = np.random.RandomState(seed); _,_,H,W = gt_f.shape
    got, tries = 0, 0
    while got < args.n and tries < args.n*12:
        tries += 1
        y,x = rng.randint(0,H-512), rng.randint(0,W-512)
        gt = gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item() < 0.045: continue
        got += 1
        crops.append((lq_f[:,:,y:y+512,x:x+512].clone(), gt.clone()))
print(f"验证 crop {len(crops)}")

import pyiqa
M = {"lpips":pyiqa.create_metric("lpips",device=dev),"dists":pyiqa.create_metric("dists",device=dev),
     "niqe":pyiqa.create_metric("niqe",device=dev),"musiq":pyiqa.create_metric("musiq",device=dev),
     "maniqa":pyiqa.create_metric("maniqa-pipal",device=dev),"clipiqa":pyiqa.create_metric("clipiqa",device=dev),
     "psnr":pyiqa.create_metric("psnr",device=dev,test_y_channel=True,color_space="ycbcr")}

def ev(fn, tag):
    res={k:[] for k in M}
    for lq, gt in crops:
        out = fn(lq); g = gt.to(dev)
        for k,f in M.items():
            v = f(out,g) if k in ("lpips","dists","psnr") else f(out)
            res[k].append(v.item())
        torch.cuda.empty_cache()
    m={k:float(np.mean(v)) for k,v in res.items()}
    m["Score"]=((1-m["lpips"])+(1-m["dists"])+m["clipiqa"]+m["maniqa"]
                +m["musiq"]/100+max(0,(10-m["niqe"])/10))
    print(f"  {tag:<24}"+"".join(f"{m[k]:>10.4f}" for k in
          ["lpips","dists","clipiqa","maniqa","musiq","niqe","psnr","Score"]))
    return m

print()
print("  "+" "*24+"".join(f"{h:>10}" for h in ["lpips","dists","clipiqa","maniqa","musiq","niqe","psnr","Score"]))
print("  "+"-"*104)
r={}
r["raw"]=ev(raw,"TinySR 训练后")
r["adain"]=ev(lambda l: fixed(l,"adain"),"TinySR训练后+adain")
r["wavelet"]=ev(lambda l: fixed(l,"wavelet"),"TinySR训练后+wavelet")
json.dump(r, open(f"{ROOT}/eval_out/tinysr_trained.json","w"), indent=1)
