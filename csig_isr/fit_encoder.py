"""找能替代 VAE encoder 的最小卷积网络。

1x1 conv 线性拟合 R^2=-0.589 失败 -> VAE encoder 是强非线性。
本实验测几种容量的小网络，找"够用且够快"的拐点。

注意：这里只做【蒸馏拟合】，看能否复现 VAE latent。
最终是否可用还要看端到端 Score（latent 微小偏差可能被 DiT 放大）。
"""
import os, sys, glob, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0,TS); os.chdir(TS)
from models.vae.autoencoder_tiny import AutoencoderTiny
dev,dt=torch.device("cuda"),torch.float16

vae=AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable",torch_dtype=dt).to(dev).eval()
for p in vae.parameters(): p.requires_grad_(False)

def make(tag, mid=0, depth=0):
    """PixelUnshuffle(8) -> [Conv3x3+SiLU]*depth -> Conv1x1(->16)"""
    layers=[nn.PixelUnshuffle(8)]
    c=192
    for _ in range(depth):
        layers += [nn.Conv2d(c, mid, 3, padding=1), nn.SiLU()]
        c=mid
    layers += [nn.Conv2d(c, 16, 1)]
    return tag, nn.Sequential(*layers)

cands=[make("1x1 conv",0,0), make("1层 c32",32,1), make("1层 c64",64,1),
       make("2层 c64",64,2), make("3层 c64",64,3), make("2层 c128",128,2)]

# 数据
fs=sorted(glob.glob(f"{ROOT}/AdcSR/testset/DIV2K-Val/HR/*.png"))[:400]
imgs=[]
for f in fs:
    a=np.asarray(Image.open(f).convert("RGB"),dtype=np.float32)/255.
    if a.shape[0]<512 or a.shape[1]<512: continue
    imgs.append(torch.from_numpy(a[:512,:512]).permute(2,0,1))
X=torch.stack(imgs)   # N,3,512,512  in [0,1]
print(f"训练样本 {X.shape[0]} 张")

def bench(m,inp,n=100,warm=30):
    with torch.no_grad():
        for _ in range(warm): m(inp)
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): m(inp)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

probe=torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
t_vae=min(bench(lambda a: vae.encode(a),probe) for _ in range(3))
print(f"VAE encoder 基准 {t_vae:.3f} ms\n")
print(f"{'结构':<14}{'参数':>9}{'时延ms':>9}{'加速':>8}{'拟合R²':>10}{'相对误差':>10}")
print("-"*62)

for tag, net in cands:
    net=net.to(dev).float().train()
    opt=torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0)
    N=X.shape[0]; bs=8
    for step in range(600):
        idx=torch.randint(0,N,(bs,))
        xb=X[idx].to(dev)
        with torch.no_grad():
            zt=vae.encode((xb*2-1).half()).latents.float()
        zp=net((xb*2-1))
        loss=F.mse_loss(zp,zt)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        xb=X[:32].to(dev)
        zt=vae.encode((xb*2-1).half()).latents.float()
        zp=net(xb*2-1)
        r2=1-((zp-zt)**2).sum()/((zt-zt.mean())**2).sum()
        rel=(zp-zt).abs().mean()/zt.abs().mean()
    nh=net.to(dev).half()
    t=min(bench(nh,probe) for _ in range(3))
    npar=sum(p.numel() for p in net.parameters())
    print(f"{tag:<14}{npar/1e3:>8.1f}K{t:>9.3f}{t_vae/t:>7.1f}x{r2.item():>10.4f}{rel.item()*100:>9.2f}%")
