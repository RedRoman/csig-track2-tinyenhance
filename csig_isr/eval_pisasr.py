"""PiSA-SR x1 适配 + 双 LoRA 系数扫描。

PiSA-SR 的独特价值：pix(保真) / sem(语义生成) 两个 LoRA 强度是【推理时超参】，
不用重训就能扫描 —— 正好对症"重度退化区幻觉"问题：调低 sem 即可。

架构：SD2.1-base UNet + 双 LoRA，单步，输入输出同尺寸（天然 x1）。
"""
import os, sys, json, itertools
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
SD21 = f"{ROOT}/sd21base"
PB = f"{ROOT}/bases/pisasr"
VAL = f"{ROOT}/data/赛题二/验证集"
dev, dt = torch.device("cuda"), torch.float16

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from safetensors.torch import load_file


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1)[None]


print("载入 PiSA-SR ...", flush=True)
unet = UNet2DConditionModel.from_pretrained(f"{PB}/unet_pix", torch_dtype=torch.float32)
vae = AutoencoderKL.from_pretrained(SD21, subfolder="vae", torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained(SD21, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(SD21, subfolder="text_encoder", torch_dtype=torch.float32)
sched = DDPMScheduler.from_pretrained(SD21, subfolder="scheduler")
sched.set_timesteps(1, device=dev)

lora_pix = load_file(f"{PB}/lora/pisa_pix.safetensors")
lora_sem = load_file(f"{PB}/lora/pisa_sem.safetensors")
print(f"  pix {len(lora_pix)} 张量, sem {len(lora_sem)} 张量", flush=True)

# 把 LoRA 按 lambda 融进 UNet 权重（A@B * scale 加到原权重）
base_sd = {k: v.clone() for k, v in unet.state_dict().items()}


def apply_lora(l_pix, l_sem):
    sd = {k: v.clone() for k, v in base_sd.items()}
    for lora, lam in [(lora_pix, l_pix), (lora_sem, l_sem)]:
        if lam == 0:
            continue
        for k in lora:
            if not k.endswith("lora_A.weight"):
                continue
            base_k = k.replace(".lora_A.weight", ".weight")
            kb = k.replace("lora_A", "lora_B")
            if base_k not in sd or kb not in lora:
                continue
            A, B = lora[k].float(), lora[kb].float()
            if A.dim() == 4:
                # A: (r, in, kh, kw)  B: (out, r, 1, 1)
                # ΔW[o,i,h,w] = Σ_r B[o,r] * A[r,i,h,w]
                d = torch.einsum("or,rihw->oihw", B.squeeze(-1).squeeze(-1), A)
            else:
                d = B @ A
            if d.shape != sd[base_k].shape:
                continue
            sd[base_k] += lam * d.to(sd[base_k].dtype)
    unet.load_state_dict(sd)


vae = vae.to(dev, dtype=dt).eval()
te = te.to(dev, dtype=dt).eval()
sched.alphas_cumprod = sched.alphas_cumprod.to(dev)
ts = torch.tensor([1], device=dev).long()   # 官方 timesteps1 默认值
with torch.no_grad():
    ids = tok([""], max_length=tok.model_max_length, padding="max_length",
              truncation=True, return_tensors="pt").input_ids.to(dev)
    null_emb = te(ids)[0].to(dt)


@torch.no_grad()
def infer(lq01):
    """PiSA-SR 官方推理：残差预测 x_denoised = z - model_pred（非 scheduler.step）。
    见 repos/PiSA-SR/pisasr.py: forward() 第 14 行。"""
    x = (lq01.to(dev, dtype=dt)*2-1)
    z = vae.encode(x).latent_dist.sample()*vae.config.scaling_factor
    m = unet(z, ts, encoder_hidden_states=null_emb).sample
    zo = z - m
    out = vae.decode(zo/vae.config.scaling_factor).sample.clamp(-1, 1)
    return (out.float()*0.5+0.5).clamp(0,1)


# 分层验证集
PAIRS = [("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
         ("case3_lq.jpg","case3_gt.jpg",303)]
items=[]
for lqn, gtn, seed in PAIRS:
    lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
    if lq_f.shape != gt_f.shape: continue
    rng = np.random.RandomState(seed); _,_,H,W = gt_f.shape
    got,tries=0,0
    while got<8 and tries<120:
        tries+=1
        y,x = rng.randint(0,H-512), rng.randint(0,W-512)
        gt = gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item()<0.035: continue
        got+=1
        lq = lq_f[:,:,y:y+512,x:x+512]
        items.append({"lq":lq.clone(),"gt":gt.clone(),
                      "psnr":-10*np.log10(F.mse_loss(lq,gt).item()+1e-12)})
ps=sorted(i["psnr"] for i in items); lo,hi=ps[len(ps)//3], ps[2*len(ps)//3]
for it in items: it["tier"]= "重" if it["psnr"]<lo else ("轻" if it["psnr"]>hi else "中")
print(f"验证集 {len(items)} crop（重/中/轻 分界 {lo:.2f}/{hi:.2f}）", flush=True)

import pyiqa
M = {"lpips":pyiqa.create_metric("lpips",device=dev),"dists":pyiqa.create_metric("dists",device=dev),
     "niqe":pyiqa.create_metric("niqe",device=dev),"musiq":pyiqa.create_metric("musiq",device=dev),
     "maniqa":pyiqa.create_metric("maniqa-pipal",device=dev),"clipiqa":pyiqa.create_metric("clipiqa",device=dev),
     "psnr":pyiqa.create_metric("psnr",device=dev,test_y_channel=True,color_space="ycbcr")}


def evaluate(tag):
    res={k:[] for k in M}; tier={"重":[],"中":[],"轻":[]}
    for it in items:
        out = infer(it["lq"]); g = it["gt"].to(dev)
        vals={}
        for k,f in M.items():
            vals[k] = (f(out,g) if k in ("lpips","dists","psnr") else f(out)).item()
            res[k].append(vals[k])
        sc=((1-vals["lpips"])+(1-vals["dists"])+vals["clipiqa"]+vals["maniqa"]
            +vals["musiq"]/100+max(0,(10-vals["niqe"])/10))
        tier[it["tier"]].append(sc)
        del out,g; torch.cuda.empty_cache()
    m={k:float(np.mean(v)) for k,v in res.items()}
    m["Score"]=((1-m["lpips"])+(1-m["dists"])+m["clipiqa"]+m["maniqa"]
                +m["musiq"]/100+max(0,(10-m["niqe"])/10))
    t={k:(float(np.mean(v)) if v else 0) for k,v in tier.items()}
    print(f"  {tag:<18}{m['Score']:>8.4f}{m['psnr']:>9.2f}"
          f"{t['重']:>9.3f}{t['中']:>9.3f}{t['轻']:>9.3f}", flush=True)
    return m, t


print()
print(f"  {'(λpix, λsem)':<18}{'Score':>8}{'PSNR':>9}{'重':>9}{'中':>9}{'轻':>9}")
print("  "+"-"*62)
out={}
for lp, ls in [(0.0,2.25)]:
    apply_lora(lp, ls)
    unet.to(dev, dtype=dt).eval()
    m,t = evaluate(f"pix{lp} sem{ls}")
    out[f"{lp}_{ls}"]={"metrics":m,"tiers":t}
    unet.to("cpu", dtype=torch.float32)
# 时延实测（与其它 base 同口径）
def bench(n=30, warm=10):
    x = torch.rand(1,3,512,512, device=dev, dtype=dt)
    with torch.no_grad():
        for _ in range(warm): infer(x)
        torch.cuda.synchronize()
        s_,e_ = torch.cuda.Event(True), torch.cuda.Event(True); s_.record()
        for _ in range(n): infer(x)
        e_.record(); torch.cuda.synchronize()
    return s_.elapsed_time(e_)/n

apply_lora(0.0, 2.25); unet.to(dev, dtype=dt).eval()
lat = bench()
sc = out["0.0_2.25"]["metrics"]["Score"]
sp = 212.768/lat
n_p = sum(p.numel() for p in unet.parameters())
print()
print(f"  PiSA-SR 时延 {lat:.2f} ms  参数量 {n_p/1e6:.1f} M")
print(f"  加速比 {sp:.2f}x   FinalScore {2**sc*sp**0.2:.2f}")
out["latency"] = lat
json.dump(out, open(f"{ROOT}/eval_out/pisasr_sweep.json","w"), indent=1)
print(f"\n已存 eval_out/pisasr_sweep.json")
