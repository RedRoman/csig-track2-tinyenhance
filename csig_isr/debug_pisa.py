"""定位 PiSA-SR 为何输出≈输入：逐步验证权重合成 -> 前向 -> timestep。"""
import os, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"; SD21=f"{ROOT}/sd21base"; PB=f"{ROOT}/bases/pisasr"
dev,dt = torch.device("cuda"), torch.float16
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from safetensors.torch import load_file

unet = UNet2DConditionModel.from_pretrained(f"{PB}/unet_pix", torch_dtype=torch.float32)
vae = AutoencoderKL.from_pretrained(SD21, subfolder="vae", torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained(SD21, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(SD21, subfolder="text_encoder", torch_dtype=torch.float32)
sched = DDPMScheduler.from_pretrained(SD21, subfolder="scheduler")
lp = load_file(f"{PB}/lora/pisa_pix.safetensors")
base_sd = {k:v.clone() for k,v in unet.state_dict().items()}

def apply_lora(lam):
    sd={k:v.clone() for k,v in base_sd.items()}
    n=0
    for k in lp:
        if not k.endswith("lora_A.weight"): continue
        bk=k.replace(".lora_A.weight",".weight"); kb=k.replace("lora_A","lora_B")
        if bk not in sd or kb not in lp: continue
        A,B=lp[k].float(),lp[kb].float()
        d = torch.einsum("or,rihw->oihw",B.squeeze(-1).squeeze(-1),A) if A.dim()==4 else B@A
        if d.shape!=sd[bk].shape: continue
        sd[bk]=sd[bk]+lam*d; n+=1
    unet.load_state_dict(sd); return n

vae=vae.to(dev,dtype=dt).eval(); te=te.to(dev,dtype=dt).eval()
sched.alphas_cumprod=sched.alphas_cumprod.to(dev)
with torch.no_grad():
    ids=tok([""],max_length=tok.model_max_length,padding="max_length",truncation=True,return_tensors="pt").input_ids.to(dev)
    emb=te(ids)[0].to(dt)

x = torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
print("=== 不同 timestep 与 lambda 下 UNet 输出的差异 ===")
print(f"{'lam':>6}{'ts':>6}{'|m|':>10}{'|z-out|':>10}{'与LQ相对差':>12}")
for lam in [0.0, 1.0]:
    n = apply_lora(lam)
    unet.to(dev,dtype=dt).eval()
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()*vae.config.scaling_factor
        for tsv in [1, 999]:
            ts = torch.tensor([tsv],device=dev).long()
            sched.set_timesteps(1, device=dev)
            m = unet(z, ts, encoder_hidden_states=emb).sample
            zo = sched.step(m, ts, z, return_dict=True).prev_sample.to(dt)
            out = vae.decode(zo/vae.config.scaling_factor).sample
            rel = (out-x).abs().mean().item()/max(x.abs().mean().item(),1e-9)
            print(f"{lam:>6.1f}{tsv:>6}{m.abs().mean().item():>10.4f}"
                  f"{(z-zo).abs().mean().item():>10.4f}{rel*100:>11.2f}%")
    unet.to("cpu",dtype=torch.float32)
print(f"\n合成层数 {n}")
print("\n判据：若 lam=0 与 lam=1 的 |m| 几乎相同 -> LoRA 未生效")
print("     若 |z-out| 极小 -> scheduler.step 在该 timestep 下几乎不改变 latent")
