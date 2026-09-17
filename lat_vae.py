"""512 单次前向的各模块时延。决定「换 decoder」是否可行。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import torch, numpy as np
from settings import TS
sys.path.insert(0, TS)
from models.vae.autoencoder_tiny import AutoencoderTiny
from diffusers import AutoencoderKL
from tinysr_runtime import build_model
DEV,DT=torch.device("cuda"),torch.float16
R="/cache/zihaohe/csig_isr"

def bench(fn, *a, n=100, w=20):
    for _ in range(w): fn(*a)
    torch.cuda.synchronize()
    ts=[]
    for _ in range(n):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(*a); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))

x=torch.randn(1,3,512,512,device=DEV,dtype=DT).clamp(-1,1)
tae=AutoencoderTiny.from_pretrained(os.path.join(TS,"checkpoint","vae","separable"),
                                    torch_dtype=DT).to(DEV,dtype=DT).eval()
kl=AutoencoderKL.from_pretrained(f"{R}/sd21base",subfolder="vae",
                                 torch_dtype=DT).to(DEV,dtype=DT).eval()
m12,_=build_model(f"{R}/train_out/run12_iqa30/best.pth",device=DEV,dtype=DT,
                  merge_lora=True,alpha_mode="fixed",fixed_alpha=1.0); m12.eval()

with torch.no_grad():
    zt=tae.encode(x).latents
    zk=kl.encode(x).latent_dist.mode()
    print(f"TAESD latent {tuple(zt.shape)}   SD2 latent {tuple(zk.shape)}")
    r={}
    r["TAESD encode"]=bench(lambda: tae.encode(x).latents)
    r["TAESD decode"]=bench(lambda: tae.decode(zt,return_dict=False)[0])
    r["SD2 encode"]  =bench(lambda: kl.encode(x).latent_dist.mode())
    r["SD2 decode"]  =bench(lambda: kl.decode(zk,return_dict=False)[0])
    r["TinySR DiT(12层)"]=bench(lambda: m12.tr(hidden_states=zt*tae.config.scaling_factor,
                                 timestep=m12.ts,pooled_projections=m12.pooled,return_dict=False)[0])
    x01=(x*0.5+0.5).clamp(0,1)
    r["TinySR 整网"]=bench(lambda: m12(x01))

print(f"\n{'模块':<22}{'中位时延 ms':>14}")
for k,v in r.items(): print(f"{k:<22}{v:>14.2f}")
tot_t=r["TAESD encode"]+r["TinySR DiT(12层)"]+r["TAESD decode"]
print(f"\n现役整网实测 {r['TinySR 整网']:.2f} ms（分项和 {tot_t:.2f}）")
sw=r["TinySR 整网"]-r["TAESD decode"]+r["SD2 decode"]
print(f"若只把 decoder 换成 SD2-VAE：{sw:.2f} ms  （加速比 {100.15/r['TinySR 整网']:.2f}× -> {100.15/sw:.2f}×）")
sw2=r["SD2 encode"]+r["TinySR DiT(12层)"]+r["SD2 decode"]
print(f"若 encoder+decoder 都换 SD2：{sw2:.2f} ms  （加速比 {100.15/sw2:.2f}×）")
