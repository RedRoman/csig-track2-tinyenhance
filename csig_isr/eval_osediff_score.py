"""测 OSEDiff 在赛题验证集上的 Score —— 用于反推 CSIG 排行榜的量纲锚点。

OSEDiff 是 x4 模型，做 x1 任务的两种用法：
  A. 512 直接进（原生 x4 -> 输出 2048，再降回 512）
  B. 512 先降到 128 -> OSEDiff x4 -> 512（常规用法，但先毁信息）
两种都测，取较优者作为"榜上 59 分平台"的估计。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
sys.path.insert(0, f"{ROOT}/OSEDiff")
os.chdir(f"{ROOT}/OSEDiff")
SD21 = f"{ROOT}/sd21base"
OSE = f"{ROOT}/AdcSR/weight/pretrained/osediff.pkl"
RAM = f"{ROOT}/AdcSR/weight/pretrained/ram_swin_large_14m.pth"
DAPE = f"{ROOT}/AdcSR/weight/pretrained/DAPE.pth"
VAL = f"{ROOT}/data/赛题二/验证集"
dev, dt = torch.device("cuda"), torch.float16
PAIRS = [("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
         ("case3_lq.jpg","case3_gt.jpg",303)]

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel

vae = AutoencoderKL.from_pretrained(SD21, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(SD21, subfolder="unet")
tok = AutoTokenizer.from_pretrained(SD21, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(SD21, subfolder="text_encoder")
sched = DDPMScheduler.from_pretrained(SD21, subfolder="scheduler")
sched.set_timesteps(1, device="cuda")
ck = torch.load(OSE, map_location="cpu", weights_only=False)
vae.load_state_dict(ck["vae"], strict=False); unet.load_state_dict(ck["unet"], strict=False)
vae = vae.to(dev,dtype=dt).eval(); unet = unet.to(dev,dtype=dt).eval(); te = te.to(dev,dtype=dt).eval()
sched.alphas_cumprod = sched.alphas_cumprod.to(dev)

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from torchvision import transforms
DAPE_m = ram(pretrained=RAM, pretrained_condition=DAPE, image_size=384, vit="swin_l").eval().to(dev).to(dt)
ram_tf = transforms.Compose([transforms.Resize((384,384)),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
ts = torch.tensor([999], device=dev).long()
print("OSEDiff 载入完成", flush=True)


@torch.no_grad()
def osediff(x01):
    """x01: [1,3,H,W] in [0,1] -> 同尺寸输出（内部走 OSEDiff 原生流程）"""
    x = (x01.to(dev,dtype=dt)*2-1)
    tag = inference(ram_tf(x01.to(dev,dtype=dt)), DAPE_m)
    p = tag[0] if isinstance(tag,(list,tuple)) else str(tag)
    ids = tok(p, max_length=tok.model_max_length, padding="max_length",
              truncation=True, return_tensors="pt").input_ids.to(dev)
    e = te(ids)[0].to(dt)
    z = vae.encode(x).latent_dist.sample()*vae.config.scaling_factor
    m = unet(z, ts, encoder_hidden_states=e).sample
    zo = sched.step(m, ts, z, return_dict=True).prev_sample.to(dt)
    out = vae.decode(zo/vae.config.scaling_factor).sample
    return (out.float()*0.5+0.5).clamp(0,1)


@torch.no_grad()
def mode_A(lq512):
    """512 直接进 OSEDiff（VAE 8x下 + 8x上，净 x1）"""
    return osediff(lq512)


@torch.no_grad()
def mode_B(lq512):
    """512 -> 降到128 -> OSEDiff -> 上采回512（模拟把 x4 模型当 SR 用）"""
    small = F.interpolate(lq512.to(dev), size=(128,128), mode="bicubic", align_corners=False).clamp(0,1)
    up = F.interpolate(small, size=(512,512), mode="bicubic", align_corners=False).clamp(0,1)
    return osediff(up)


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1)[None]

crops=[]
for lqn, gtn, seed in PAIRS:
    lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
    if lq_f.shape != gt_f.shape: continue
    rng = np.random.RandomState(seed); _,_,H,W = gt_f.shape
    got, tries = 0, 0
    while got < 6 and tries < 72:
        tries += 1
        y,x = rng.randint(0,H-512), rng.randint(0,W-512)
        gt = gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item() < 0.045: continue
        got += 1
        crops.append((lq_f[:,:,y:y+512,x:x+512].clone(), gt.clone()))
print(f"验证 crop {len(crops)}", flush=True)

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
    print(f"  {tag:<28}"+"".join(f"{m[k]:>10.4f}" for k in
          ["lpips","dists","clipiqa","maniqa","musiq","niqe","psnr","Score"]), flush=True)
    return m

print()
print("  "+" "*28+"".join(f"{h:>10}" for h in ["lpips","dists","clipiqa","maniqa","musiq","niqe","psnr","Score"]))
print("  "+"-"*108)
r={}
r["A_direct512"]=ev(mode_A,"OSEDiff A: 512直接进")
r["B_down_up"]=ev(mode_B,"OSEDiff B: 降128再x4")
json.dump(r, open(f"{ROOT}/eval_out/osediff_score.json","w"), indent=1)
print(f"\n已存 eval_out/osediff_score.json")
