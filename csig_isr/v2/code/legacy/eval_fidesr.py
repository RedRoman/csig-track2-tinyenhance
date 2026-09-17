"""FiDeSR (CVPR2026) x1 评测：SD2.1 UNet + LoRA + LRRB 精修，残差预测。

官方推理链路（repos/FiDeSR/fidesr.py forward）：
    z = vae.encode(x) * sf
    model_pred = unet(z, ts1, prompt_emb)
    delta_r = lrrb(cat([z, model_pred]))
    r_tilde = model_pred + delta_r
    x_denoised = z - r_tilde
    out = vae.decode(x_denoised / sf)
"""
import os, sys, json
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"
sys.path.insert(0, f"{ROOT}/repos/FiDeSR")
SD21=f"{ROOT}/sd21base"; VAL=f"{ROOT}/data/赛题二/验证集"
dev,dt = torch.device("cuda"), torch.float16

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig

ck = torch.load(f"{ROOT}/bases/fidesr/fidesr.pkl", map_location="cpu", weights_only=False)
print(f"rank={ck['lora_rank_unet']}, enc/dec/others "
      f"{len(ck['unet_lora_encoder_modules'])}/{len(ck['unet_lora_decoder_modules'])}/"
      f"{len(ck['unet_lora_others_modules'])}", flush=True)

# 直接用官方 LRRB 定义（避免手写出错）
import importlib.util
_spec = importlib.util.spec_from_file_location("_fidesr_mod", f"{ROOT}/repos/FiDeSR/fidesr.py")
try:
    _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
    LRRB = _m.LRRB
    print("已 import 官方 LRRB", flush=True)
except Exception as _e:
    # 官方文件顶层可能有其它依赖，退化为按源码复刻
    class DenseBlock(nn.Module):
        def __init__(self, in_ch, growth=32):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, growth, 3, 1, 1)
            self.conv2 = nn.Conv2d(in_ch + growth, growth, 3, 1, 1)
            self.conv3 = nn.Conv2d(in_ch + 2*growth, growth, 3, 1, 1)
            self.conv4 = nn.Conv2d(in_ch + 3*growth, growth, 3, 1, 1)
            self.conv5 = nn.Conv2d(in_ch + 4*growth, in_ch, 3, 1, 1)
            self.act = nn.SiLU()
        def forward(self, x):
            x1 = self.act(self.conv1(x))
            x2 = self.act(self.conv2(torch.cat([x, x1], 1)))
            x3 = self.act(self.conv3(torch.cat([x, x1, x2], 1)))
            x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], 1)))
            return self.conv5(torch.cat([x, x1, x2, x3, x4], 1))

    class LRRB(nn.Module):
        def __init__(self, in_ch=8, mid_ch=64, growth=32, res_scale=0.2):
            super().__init__()
            self.in_conv = nn.Conv2d(in_ch, mid_ch, 1, 1, 0)
            self.db1 = DenseBlock(mid_ch, growth)
            self.db2 = DenseBlock(mid_ch, growth)
            self.db3 = DenseBlock(mid_ch, growth)
            self.out_conv = nn.Conv2d(mid_ch, in_ch // 2, 1, 1, 0)
            self.res_scale = res_scale
        def forward(self, x_cat):
            f = self.in_conv(x_cat)
            f = f + self.db1(f) * self.res_scale
            f = f + self.db2(f) * self.res_scale
            f = f + self.db3(f) * self.res_scale
            return self.out_conv(f)
    print(f"官方 import 失败({str(_e)[:40]})，用复刻版", flush=True)

unet = UNet2DConditionModel.from_pretrained(SD21, subfolder="unet")
r = ck["lora_rank_unet"]
for nm, mods in [("default_encoder", ck["unet_lora_encoder_modules"]),
                 ("default_decoder", ck["unet_lora_decoder_modules"]),
                 ("default_others",  ck["unet_lora_others_modules"])]:
    unet.add_adapter(LoraConfig(r=r, init_lora_weights="gaussian", target_modules=mods),
                     adapter_name=nm)
sd_u = ck["state_dict_unet"]
cur = unet.state_dict(); hit=0
for k,v in sd_u.items():
    if k in cur and cur[k].shape==v.shape:
        cur[k].copy_(v); hit+=1
unet.load_state_dict(cur)
unet.set_adapter(["default_encoder","default_decoder","default_others"])
print(f"UNet LoRA 载入 {hit}/{len(sd_u)}", flush=True)

lrrb = LRRB(in_ch=8, mid_ch=64, growth=32, res_scale=0.2)
try:
    lrrb.load_state_dict(ck["lrrb_state_dict"]); print("LRRB 载入成功", flush=True)
except Exception as e:
    ls = ck["lrrb_state_dict"]
    print(f"LRRB 结构不符，实际键: {list(ls.keys())[:6]}", flush=True)
    shapes = {k: tuple(v.shape) for k,v in list(ls.items())[:8]}
    print(f"  形状: {shapes}", flush=True)
    sys.exit(1)

vae = AutoencoderKL.from_pretrained(SD21, subfolder="vae")
tok = AutoTokenizer.from_pretrained(SD21, subfolder="tokenizer")
te  = CLIPTextModel.from_pretrained(SD21, subfolder="text_encoder")
unet=unet.to(dev,dtype=dt).eval(); vae=vae.to(dev,dtype=dt).eval()
te=te.to(dev,dtype=dt).eval(); lrrb=lrrb.to(dev,dtype=dt).eval()
ts = torch.tensor([1], device=dev).long()
with torch.no_grad():
    ids = tok([""],max_length=tok.model_max_length,padding="max_length",
              truncation=True,return_tensors="pt").input_ids.to(dev)
    emb = te(ids)[0].to(dt)

@torch.no_grad()
def infer(lq01):
    x = lq01.to(dev,dtype=dt)*2-1
    sf = vae.config.scaling_factor
    z = vae.encode(x).latent_dist.sample()*sf
    mp = unet(z, ts, encoder_hidden_states=emb).sample.to(z.dtype)
    r_t = mp + lrrb(torch.cat([z,mp],1))
    out = vae.decode((z-r_t)/sf).sample.clamp(-1,1)
    return (out.float()*0.5+0.5).clamp(0,1)

def load_img(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1)[None]

PAIRS=[("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
       ("case3_lq.jpg","case3_gt.jpg",303)]
items=[]
for lqn,gtn,seed in PAIRS:
    lq_f,gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
    if lq_f.shape!=gt_f.shape: continue
    rng=np.random.RandomState(seed); _,_,H,W=gt_f.shape
    got,tries=0,0
    while got<8 and tries<120:
        tries+=1
        y,x=rng.randint(0,H-512),rng.randint(0,W-512)
        gt=gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item()<0.035: continue
        got+=1
        lq=lq_f[:,:,y:y+512,x:x+512]
        items.append({"lq":lq.clone(),"gt":gt.clone(),
                      "psnr":-10*np.log10(F.mse_loss(lq,gt).item()+1e-12)})
ps=sorted(i["psnr"] for i in items); lo,hi=ps[len(ps)//3],ps[2*len(ps)//3]
for it in items: it["tier"]="重" if it["psnr"]<lo else ("轻" if it["psnr"]>hi else "中")
print(f"验证集 {len(items)} crop", flush=True)

import pyiqa
M={"lpips":pyiqa.create_metric("lpips",device=dev),"dists":pyiqa.create_metric("dists",device=dev),
   "niqe":pyiqa.create_metric("niqe",device=dev),"musiq":pyiqa.create_metric("musiq",device=dev),
   "maniqa":pyiqa.create_metric("maniqa-pipal",device=dev),"clipiqa":pyiqa.create_metric("clipiqa",device=dev),
   "psnr":pyiqa.create_metric("psnr",device=dev,test_y_channel=True,color_space="ycbcr")}
res={k:[] for k in M}; tier={"重":[],"中":[],"轻":[]}
for it in items:
    out=infer(it["lq"]); g=it["gt"].to(dev); vals={}
    for k,f in M.items():
        vals[k]=(f(out,g) if k in ("lpips","dists","psnr") else f(out)).item()
        res[k].append(vals[k])
    tier[it["tier"]].append((1-vals["lpips"])+(1-vals["dists"])+vals["clipiqa"]
        +vals["maniqa"]+vals["musiq"]/100+max(0,(10-vals["niqe"])/10))
    del out,g; torch.cuda.empty_cache()
m={k:float(np.mean(v)) for k,v in res.items()}
m["Score"]=((1-m["lpips"])+(1-m["dists"])+m["clipiqa"]+m["maniqa"]
            +m["musiq"]/100+max(0,(10-m["niqe"])/10))
t={k:(float(np.mean(v)) if v else 0) for k,v in tier.items()}

def bench(n=30,warm=10):
    x=torch.rand(1,3,512,512,device=dev,dtype=dt)
    with torch.no_grad():
        for _ in range(warm): infer(x)
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): infer(x)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n
lat=bench()
print()
print("="*68)
print(f"  FiDeSR (CVPR2026)  Score {m['Score']:.4f}  PSNR {m['psnr']:.2f}  时延 {lat:.2f} ms")
print(f"    分层: 重 {t['重']:.3f}  中 {t['中']:.3f}  轻 {t['轻']:.3f}")
print(f"    lpips {m['lpips']:.4f} dists {m['dists']:.4f} clipiqa {m['clipiqa']:.4f} "
      f"maniqa {m['maniqa']:.4f} musiq {m['musiq']:.2f} niqe {m['niqe']:.4f}")
sp=212.768/lat
print(f"    加速比 {sp:.2f}x   FinalScore {2**m['Score']*sp**0.2:.2f}")
json.dump({"metrics":m,"tiers":t,"latency":lat},
          open(f"{ROOT}/eval_out/fidesr.json","w"),indent=1)
