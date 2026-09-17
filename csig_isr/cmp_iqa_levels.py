"""逐档位对比各 w_iqa 的【分项指标】，判断 iqa2.0 是"真变好"还是"刷无参考分"。

判据：
  - 若 LPIPS/DISTS（有参考，衡量与 GT 的结构/纹理相似度）也改善
    -> 说明生成的内容确实更接近 GT，不是凭空编造
  - 若只有 CLIPIQA/MUSIQ/NIQE（无参考）改善而 LPIPS/DISTS 恶化
    -> 才是"指标异常高但主观不合理"的刷分行为
"""
import os, sys, json
import numpy as np, torch
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

CANDS=[("zeroshot",None),
       ("iqa0.5",f"{ROOT}/train_out/tinysr2/best.pth"),
       ("iqa0.8",f"{ROOT}/train_out/tinysr5/best.pth"),
       ("iqa1.2",f"{ROOT}/train_out/tinysr6/best.pth"),
       ("iqa1.2+fid",f"{ROOT}/train_out/tinysr8_fid/best.pth"),
       ("iqa2.0",f"{ROOT}/train_out/tinysr7/best.pth"),
       ("bigdata",f"{ROOT}/train_out/tinysr10_bigdata/best.pth")]
CANDS=[(n,p) for n,p in CANDS if p is None or os.path.exists(p)]

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
base={k:v.clone() for k,v in tr.state_dict().items() if "lora_" in k}
sf=vae.config.scaling_factor

def set_ck(p):
    cur=tr.state_dict()
    src=base if p is None else torch.load(p,map_location="cpu",weights_only=False)["lora"]
    for k,v in src.items():
        if k in cur and cur[k].shape==v.shape: cur[k].copy_(v.to(cur[k].dtype))

@torch.no_grad()
def infer(lq01):
    x=lq01.to(dev,dtype=dt)*2-1
    z=vae.encode(x).latents*sf
    p=tr(hidden_states=z,timestep=ts,pooled_projections=pooled,return_dict=False)[0]
    img=vae.decode((z-p)/sf,return_dict=False)[0]
    sr=(img.float()*0.5+0.5).clamp(0,1)
    f=wavelet_color_fix(ToPILImage()(sr[0].cpu()),ToPILImage()(lq01[0].cpu()))
    return ToTensor()(f)[None].to(dev)

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
    while got<16 and tries<240:
        tries+=1
        y,x=rng.randint(0,H-512),rng.randint(0,W-512)
        gt=gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item()<0.035: continue
        got+=1
        lq=lq_f[:,:,y:y+512,x:x+512]
        import torch.nn.functional as F
        items.append({"lq":lq.clone(),"gt":gt.clone(),"case":lqn[:5],
                      "psnr":-10*np.log10(F.mse_loss(lq,gt).item()+1e-12)})
ps=sorted(i["psnr"] for i in items); lo,hi=ps[len(ps)//3],ps[2*len(ps)//3]
for it in items: it["tier"]="重" if it["psnr"]<lo else ("轻" if it["psnr"]>hi else "中")
print(f"验证 {len(items)} crop（重/中/轻 分界 {lo:.2f}/{hi:.2f}）\n",flush=True)

import pyiqa
M={"lpips":pyiqa.create_metric("lpips",device=dev),"dists":pyiqa.create_metric("dists",device=dev),
   "niqe":pyiqa.create_metric("niqe",device=dev),"musiq":pyiqa.create_metric("musiq",device=dev),
   "maniqa":pyiqa.create_metric("maniqa-pipal",device=dev),"clipiqa":pyiqa.create_metric("clipiqa",device=dev),
   "psnr":pyiqa.create_metric("psnr",device=dev,test_y_channel=True,color_space="ycbcr")}

hdr=f"{'档位':<14}{'LPIPS↓':>9}{'DISTS↓':>9}{'CLIPIQA↑':>10}{'MANIQA↑':>9}{'MUSIQ↑':>8}{'NIQE↓':>8}{'PSNR':>8}{'Score':>8}"
print(hdr); print("-"*len(hdr))
res_all={}
for name,ck in CANDS:
    set_ck(ck)
    r={k:[] for k in M}; tier={"重":[],"中":[],"轻":[]}
    for it in items:
        o=infer(it["lq"]); g=it["gt"].to(dev); v={}
        for k,f in M.items():
            v[k]=(f(o,g) if k in ("lpips","dists","psnr") else f(o)).item(); r[k].append(v[k])
        tier[it["tier"]].append((1-v["lpips"])+(1-v["dists"])+v["clipiqa"]+v["maniqa"]
                                +v["musiq"]/100+max(0,(10-v["niqe"])/10))
        torch.cuda.empty_cache()
    m={k:float(np.mean(x)) for k,x in r.items()}
    m["Score"]=((1-m["lpips"])+(1-m["dists"])+m["clipiqa"]+m["maniqa"]+m["musiq"]/100
                +max(0,(10-m["niqe"])/10))
    m["tiers"]={k:(float(np.mean(v)) if v else 0) for k,v in tier.items()}
    res_all[name]=m
    print(f"{name:<14}{m['lpips']:>9.4f}{m['dists']:>9.4f}{m['clipiqa']:>10.4f}"
          f"{m['maniqa']:>9.4f}{m['musiq']:>8.2f}{m['niqe']:>8.3f}{m['psnr']:>8.2f}{m['Score']:>8.4f}",flush=True)

print()
print("="*80)
print("关键判据：iqa2.0 相对 iqa1.2+fid 的分项变化")
print("="*80)
a=res_all.get("iqa1.2+fid"); b=res_all.get("iqa2.0")
if a and b:
    print(f"  {'指标':<12}{'iqa1.2+fid':>12}{'iqa2.0':>12}{'变化':>12}{'方向':>10}")
    for k,better_low in [("lpips",True),("dists",True),("clipiqa",False),
                         ("maniqa",False),("musiq",False),("niqe",True),("psnr",False)]:
        d=b[k]-a[k]
        good = (d<0) if better_low else (d>0)
        print(f"  {k:<12}{a[k]:>12.4f}{b[k]:>12.4f}{d:>+12.4f}{'改善' if good else '恶化':>10}")
    print()
    print(f"  有参考项(LPIPS+DISTS)贡献: {(1-a['lpips'])+(1-a['dists']):.4f} -> "
          f"{(1-b['lpips'])+(1-b['dists']):.4f}  ({((1-b['lpips'])+(1-b['dists']))-((1-a['lpips'])+(1-a['dists'])):+.4f})")
    print(f"  无参考四项贡献:            "
          f"{a['clipiqa']+a['maniqa']+a['musiq']/100+max(0,(10-a['niqe'])/10):.4f} -> "
          f"{b['clipiqa']+b['maniqa']+b['musiq']/100+max(0,(10-b['niqe'])/10):.4f}")
    print()
    print("  分层 Score:")
    for t in ["重","中","轻"]:
        print(f"    {t}: {a['tiers'][t]:.4f} -> {b['tiers'][t]:.4f}  ({b['tiers'][t]-a['tiers'][t]:+.4f})")
json.dump(res_all,open(f"{ROOT}/eval_out/iqa_levels.json","w"),indent=1,ensure_ascii=False)
