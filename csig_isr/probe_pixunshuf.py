"""验证 VAE encoder 换 PixelUnshuffle 的可行性。

AdcSR 的做法：把 VAE encoder 整个删掉，用 PixelUnshuffle 直接把图像重排成 latent。
实测 -9% 参数、-40% 时间，且 PSNR 反而 +0.13dB（VAE encoder 本身有信息损失）。

TinySR 的 latent 是 16 通道 64x64（512 输入），
PixelUnshuffle(8) 给出 3*64=192 通道 64x64 —— 通道数不匹配，
需要一个 1x1 conv 做 192->16 的投影（可用 VAE encoder 的输出统计初始化）。
"""
import os, sys, torch, torch.nn as nn
import numpy as np
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0,TS); os.chdir(TS)
from models.vae.autoencoder_tiny import AutoencoderTiny
dev,dt=torch.device("cuda"),torch.float16

vae=AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable",torch_dtype=dt).to(dev).eval()
print(f"latent_channels = {vae.config.latent_channels}, scaling = {vae.config.scaling_factor}")
n_enc=sum(p.numel() for p in vae.encoder.parameters())
n_dec=sum(p.numel() for p in vae.decoder.parameters())
print(f"encoder {n_enc/1e6:.3f}M   decoder {n_dec/1e6:.3f}M")

x=torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
with torch.no_grad():
    z=vae.encode(x).latents
print(f"encode: {tuple(x.shape)} -> {tuple(z.shape)}")
print(f"  latent 统计 mean {z.float().mean():.4f} std {z.float().std():.4f} "
      f"range [{z.float().min():.3f}, {z.float().max():.3f}]")

pu=nn.PixelUnshuffle(8)
with torch.no_grad():
    pz=pu(x)
print(f"PixelUnshuffle(8): {tuple(x.shape)} -> {tuple(pz.shape)}")
print(f"  统计 mean {pz.float().mean():.4f} std {pz.float().std():.4f}")

print()
print("方案：PixelUnshuffle(8) + 1x1 conv (192->16)")
proj=nn.Conv2d(192,16,1).to(dev,dtype=dt)
n_proj=sum(p.numel() for p in proj.parameters())
print(f"  投影层参数 {n_proj/1e3:.1f}K  (vs VAE encoder {n_enc/1e6:.3f}M)")

def bench(f,inp,n=100,warm=30):
    with torch.no_grad():
        for _ in range(warm): f(inp)
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): f(inp)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

t_vae=min(bench(lambda a: vae.encode(a),x) for _ in range(3))
t_pu =min(bench(lambda a: proj(pu(a)),x) for _ in range(3))
print(f"\n  VAE encoder      {t_vae:>7.3f} ms")
print(f"  PixelUnshuffle+proj {t_pu:>7.3f} ms   加速 {t_vae/t_pu:.1f}x  省 {t_vae-t_pu:.2f} ms")

# 关键：能否用最小二乘拟合出 proj，使其近似 VAE encoder
print()
print("拟合可行性：用 512 张随机 crop 做最小二乘，看 PixelUnshuffle+proj 能否逼近 VAE latent")
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
fs=sorted(glob.glob(f"{ROOT}/AdcSR/testset/DIV2K-Val/HR/*.png"))[:64]
X=[];Y=[]
with torch.no_grad():
    for f in fs:
        a=np.asarray(Image.open(f).convert("RGB"),dtype=np.float32)/255.
        t=torch.from_numpy(a).permute(2,0,1)[None].to(dev,dtype=dt)*2-1
        if t.shape[-1]<512: continue
        t=t[:,:,:512,:512]
        X.append(pu(t).float().permute(0,2,3,1).reshape(-1,192).cpu())
        Y.append(vae.encode(t).latents.float().permute(0,2,3,1).reshape(-1,16).cpu())
X=torch.cat(X); Y=torch.cat(Y)
print(f"  样本 {X.shape[0]} 个像素点, 输入 {X.shape[1]} 维 -> 输出 {Y.shape[1]} 维")
Xa=torch.cat([X, torch.ones(X.shape[0],1)],1)
W,_,_,_=torch.linalg.lstsq(Xa, Y)
pred=Xa@W
r2=1-((pred-Y)**2).sum()/((Y-Y.mean(0))**2).sum()
print(f"  线性拟合 R^2 = {r2:.4f}")
print(f"  {'-> 线性可分，1x1 conv 足够' if r2>0.9 else '-> 线性不足，需要非线性层（几层卷积）'}")
