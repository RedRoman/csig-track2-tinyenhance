"""最干净的单块测量：512 中心块，不 tile 不融合，排除拼接干扰。
自重建 PSNR 是编码器信息损失的直接读数。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from settings import VAL, TS
from tinysr_runtime import build_model
sys.path.insert(0, TS)
from models.vae.autoencoder_tiny import AutoencoderTiny
from diffusers import AutoencoderKL
Image.MAX_IMAGE_PIXELS=None
DEV,DT = torch.device("cuda"), torch.float16
R="/cache/zihaohe/csig_isr"
def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]
def crop(t):
    H,W=t.shape[-2:]; return t[:,:,H//2-256:H//2+256, W//2-256:W//2+256].contiguous()
def psnr(a,b): return 10*np.log10(1.0/max(float(((a-b)**2).mean()),1e-12))

tae = AutoencoderTiny.from_pretrained(os.path.join(TS,"checkpoint","vae","separable"),
                                      torch_dtype=DT).to(DEV,dtype=DT).eval()
kl  = AutoencoderKL.from_pretrained(f"{R}/sd21base", subfolder="vae",
                                    torch_dtype=DT).to(DEV,dtype=DT).eval()
m12,_ = build_model(f"{R}/train_out/run12_iqa30/best.pth", device=DEV, dtype=DT,
                    merge_lora=False, alpha_mode="fixed", fixed_alpha=1.0); m12.eval()

@torch.no_grad()
def tae_rt(x, vae):
    sf=vae.config.scaling_factor
    z=vae.encode(x*2-1).latents*sf
    return (vae.decode(z/sf, return_dict=False)[0]*0.5+0.5).clamp(0,1).float()
@torch.no_grad()
def kl_rt(x):
    z=kl.encode(x*2-1).latent_dist.mode()
    return (kl.decode(z, return_dict=False)[0]*0.5+0.5).clamp(0,1).float()

print(f"{'case':<7}{'GT→TAE→ vs GT':>16}{'LQ→TAE→ vs LQ':>16}{'GT→SD2→ vs GT':>16}"
      f"{'LQ→SD2→ vs LQ':>16}{'run12(LQ) vs GT':>18}{'LQ vs GT':>11}")
for i,(l,g) in enumerate([("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),
                          ("case3_lq.jpg","case3_gt.jpg")],1):
    lq=crop(load(os.path.join(VAL,l))).to(DEV,dtype=DT)
    gt=crop(load(os.path.join(VAL,g))).to(DEV,dtype=DT)
    lqf, gtf = lq.float(), gt.float()
    a=psnr(tae_rt(gt,tae), gtf); b=psnr(tae_rt(lq,tae), lqf)
    c=psnr(kl_rt(gt), gtf);      d=psnr(kl_rt(lq), lqf)
    with torch.no_grad(), torch.autocast(device_type="cuda",dtype=DT):
        o=m12(lq).float().clamp(0,1)
    e=psnr(o,gtf); f=psnr(lqf,gtf)
    print(f"case{i:<3}{a:>16.2f}{b:>16.2f}{c:>16.2f}{d:>16.2f}{e:>18.2f}{f:>11.2f}")
# run12 的 decoder 与出厂 TAESD 是否一致
d0=dict(tae.decoder.named_parameters()); d1=dict(m12.vae.decoder.named_parameters())
diff=max(float((d1[k].float()-d0[k].float()).abs().max()) for k in d0 if k in d1)
print(f"\nrun12 的 vae.decoder 与出厂 TAESD 最大参数差 = {diff:.3e}")
