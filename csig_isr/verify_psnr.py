"""验证 LQ 基线 PSNR 28.99 是否正确，并量化 perception-distortion tradeoff。"""
import os, sys, json
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"; VAL=f"{ROOT}/data/赛题二/验证集"
dev=torch.device("cuda")

def load_img(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1)[None]

PAIRS=[("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
       ("case3_lq.jpg","case3_gt.jpg",303)]
crops=[]
for lqn,gtn,seed in PAIRS:
    lq_f,gt_f=load_img(f"{VAL}/{lqn}"),load_img(f"{VAL}/{gtn}")
    if lq_f.shape!=gt_f.shape: continue
    rng=np.random.RandomState(seed); _,_,H,W=gt_f.shape
    got,tries=0,0
    while got<6 and tries<72:
        tries+=1
        y,x=rng.randint(0,H-512),rng.randint(0,W-512)
        gt=gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item()<0.045: continue
        got+=1
        crops.append((lqn[:5],lq_f[:,:,y:y+512,x:x+512].clone(),gt.clone()))
print(f"crop {len(crops)}")

import pyiqa
psnr_y = pyiqa.create_metric("psnr",device=dev,test_y_channel=True,color_space="ycbcr")
psnr_rgb = pyiqa.create_metric("psnr",device=dev,test_y_channel=False)
ssim = pyiqa.create_metric("ssim",device=dev,test_y_channel=True,color_space="ycbcr")

print()
print("=== 三种 PSNR 口径对比（LQ vs GT）===")
a_y=[];a_rgb=[];a_np=[];a_ss=[]
for tag,lq,gt in crops:
    l,g=lq.to(dev),gt.to(dev)
    a_y.append(psnr_y(l,g).item())
    a_rgb.append(psnr_rgb(l,g).item())
    a_np.append(-10*np.log10(F.mse_loss(l,g).item()))
    a_ss.append(ssim(l,g).item())
print(f"  pyiqa PSNR (Y通道/YCbCr)   {np.mean(a_y):.3f} dB   <- score_eval.py 用的口径")
print(f"  pyiqa PSNR (RGB)           {np.mean(a_rgb):.3f} dB")
print(f"  手算 MSE->PSNR (RGB)       {np.mean(a_np):.3f} dB   <- train 里 val_real 用的口径")
print(f"  SSIM (Y通道)               {np.mean(a_ss):.4f}")
print()
print("  差异来源：Y 通道只看亮度，退化主要在亮度高频，但 Y 通道会加权(0.299R+0.587G+0.114B)")
print("            且色度通道退化更轻，故 Y 通道 PSNR 高于 RGB")
