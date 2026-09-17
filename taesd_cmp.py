"""TinySR 的 separable TAESD  vs  官方 taesd3。config 完全一致，只差权重。"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from settings import VAL, TS
sys.path.insert(0, TS)
from models.vae.autoencoder_tiny import AutoencoderTiny as TinyTS
from diffusers import AutoencoderTiny as TinyHF
Image.MAX_IMAGE_PIXELS=None
DEV,DT=torch.device("cuda"),torch.float16
OFF = glob.glob('/cache/zihaohe/hf/hub/models--madebyollin--taesd3/snapshots/*')[0]
SEP = os.path.join(TS,"checkpoint","vae","separable")

def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]
def crop(t,s=512):
    H,W=t.shape[-2:]; return t[:,:,H//2-s//2:H//2+s//2, W//2-s//2:W//2+s//2].contiguous()
def psnr(a,b): return 10*np.log10(1.0/max(float(((a-b)**2).mean()),1e-12))
def bench(fn,n=60,w=15):
    with torch.no_grad():
        for _ in range(w): fn()
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            s=torch.cuda.Event(True);e=torch.cuda.Event(True)
            s.record();fn();e.record();torch.cuda.synchronize();ts.append(s.elapsed_time(e))
    return float(np.median(ts))

sep = TinyTS.from_pretrained(SEP, torch_dtype=DT).to(DEV,dtype=DT).eval()
off = TinyHF.from_pretrained(OFF, torch_dtype=DT).to(DEV,dtype=DT).eval()
# 官方权重能否直接装进 TinySR 的类
sep2 = TinyTS.from_pretrained(SEP, torch_dtype=DT).to(DEV,dtype=DT).eval()
rep = sep2.load_state_dict(off.state_dict(), strict=False)
print(f"官方 taesd3 权重装进 TinySR 的 AutoencoderTiny 类：缺失 {len(rep.missing_keys)} 多余 {len(rep.unexpected_keys)}")
ds, do = sep.state_dict(), off.state_dict()
shared=[k for k in ds if k in do and ds[k].shape==do[k].shape]
print(f"同名同形状张量 {len(shared)}/{len(ds)}；权重最大差 "
      f"{max(float((ds[k].float()-do[k].float()).abs().max()) for k in shared):.4f}")

@torch.no_grad()
def rt(v,x):
    sf=v.config.scaling_factor
    z=v.encode(x*2-1).latents*sf
    return (v.decode(z/sf,return_dict=False)[0]*0.5+0.5).clamp(0,1).float()

print(f"\n{'图':<10}{'TinySR separable':>20}{'官方 taesd3':>18}{'差':>9}")
for i,g in enumerate(["case1_gt.png","case2_gt.png","case3_gt.jpg"],1):
    x=crop(load(os.path.join(VAL,g))).to(DEV,dtype=DT); xf=x.float()
    a,b=psnr(rt(sep,x),xf), psnr(rt(off,x),xf)
    print(f"case{i}_gt{'':<3}{a:>20.2f}{b:>18.2f}{b-a:>9.2f}")
for i,l in enumerate(["case1_lq.jpg","case2_lq.jpg","case3_lq.jpg"],1):
    x=crop(load(os.path.join(VAL,l))).to(DEV,dtype=DT); xf=x.float()
    a,b=psnr(rt(sep,x),xf), psnr(rt(off,x),xf)
    print(f"case{i}_lq{'':<3}{a:>20.2f}{b:>18.2f}{b-a:>9.2f}")

x=crop(load(os.path.join(VAL,"case2_gt.png"))).to(DEV,dtype=DT)
with torch.no_grad():
    zs=sep.encode(x*2-1).latents; zo=off.encode(x*2-1).latents
print(f"\nlatent 统计  separable mean={zs.float().mean():+.4f} std={zs.float().std():.4f}"
      f"   官方 mean={zo.float().mean():+.4f} std={zo.float().std():.4f}")
print(f"两者 latent 相关系数 {float(torch.corrcoef(torch.stack([zs.flatten().float(),zo.flatten().float()]))[0,1]):+.4f}")
print(f"\n时延  separable enc {bench(lambda: sep.encode(x*2-1).latents):.2f} ms"
      f"  dec {bench(lambda: sep.decode(zs,return_dict=False)[0]):.2f} ms")
print(f"     官方taesd3 enc {bench(lambda: off.encode(x*2-1).latents):.2f} ms"
      f"  dec {bench(lambda: off.decode(zo,return_dict=False)[0]):.2f} ms")
