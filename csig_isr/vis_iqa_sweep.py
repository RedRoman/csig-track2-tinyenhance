"""不同 w_iqa 训练结果的视觉对比 —— 判断锐化是否过头。

赛题红线：「指标异常高但主观明显不合理 -> 取消资格」，入围要人工复核。
所以每个候选都必须肉眼确认，不能只看 Score。
"""
import os, sys, glob
import numpy as np
import torch
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
TS = f"{ROOT}/TinySR"
sys.path.insert(0, TS); os.chdir(TS)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from utils.wavelet_color_fix import wavelet_color_fix
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
from torchvision.transforms import ToPILImage, ToTensor

VAL = f"{ROOT}/data/赛题二/验证集"
OUT = f"{ROOT}/visualizations/iqa_sweep"
os.makedirs(OUT, exist_ok=True)
dev, dt = torch.device("cuda"), torch.float16

CANDS = [("zeroshot", None),
         ("iqa0.5", f"{ROOT}/train_out/tinysr2/best.pth"),
         ("iqa0.8", f"{ROOT}/train_out/tinysr5/best.pth"),
         ("iqa1.2", f"{ROOT}/train_out/tinysr6/best.pth"),
         ("iqa2.0", f"{ROOT}/train_out/tinysr7/best.pth")]
CANDS = [(n,p) for n,p in CANDS if p is None or os.path.exists(p)]


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1)[None]


def build():
    tr = TinySD3Transformer2DModel.from_pretrained(
        f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr", subfolder="transformer",
        torch_dtype=dt, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    vae = AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable", torch_dtype=dt)
    lc = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian",
                    target_modules=["to_k","to_q","to_v","to_out.0","proj","linear","linear_1","linear_2","net.2"])
    tr.add_adapter(lc); tr.enable_adapters()
    load_lora_state_dict(StableDiffusion3Pipeline.lora_state_dict(
        f"{TS}/checkpoint/tinysr", weight_name="transformer.safetensors"), tr)
    vae = vae.to(dev,dtype=dt).eval(); tr = tr.to(dev,dtype=dt).eval()
    ts = torch.tensor([1000.0], device=dev, dtype=dt)
    pooled = torch.load(f"{TS}/dataset/default/pool_embeds.pt", map_location=dev).to(dtype=dt)
    with torch.no_grad():
        w = torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
        zz = vae.encode(w).latents*vae.config.scaling_factor
        _ = tr(hidden_states=zz, timestep=ts, pooled_projections=pooled, return_dict=False)
    return vae, tr, ts, pooled


vae, tr, ts, pooled = build()
base_lora = {k: v.clone() for k, v in tr.state_dict().items() if "lora_" in k}


def set_ckpt(path):
    cur = tr.state_dict()
    src = base_lora if path is None else torch.load(path, map_location="cpu", weights_only=False)["lora"]
    for k, v in src.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k].copy_(v.to(cur[k].dtype))


@torch.no_grad()
def infer(lq01):
    x = lq01.to(dev,dtype=dt)*2-1
    z = vae.encode(x).latents*vae.config.scaling_factor
    p = tr(hidden_states=z, timestep=ts, pooled_projections=pooled, return_dict=False)[0]
    img = vae.decode((z-p)/vae.config.scaling_factor, return_dict=False)[0]
    sr = (img.float()*0.5+0.5).clamp(0,1)
    f = wavelet_color_fix(ToPILImage()(sr[0].cpu()), ToPILImage()(lq01[0].cpu()))
    return ToTensor()(f)[None]


# 取 3 个高纹理 crop
PAIRS = [("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
         ("case3_lq.jpg","case3_gt.jpg",303)]
crops=[]
for lqn, gtn, seed in PAIRS:
    lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
    if lq_f.shape != gt_f.shape: continue
    rng = np.random.RandomState(seed); _,_,H,W = gt_f.shape
    got,tries=0,0
    while got<1 and tries<40:
        tries+=1
        y,x = rng.randint(0,H-512), rng.randint(0,W-512)
        gt = gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item()<0.07: continue
        got+=1
        crops.append((lqn[:5], lq_f[:,:,y:y+512,x:x+512].clone(), gt.clone()))
print(f"crop {len(crops)}")

def sv(t,p):
    a=(t[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
    Image.fromarray(a).save(p)

cols = ["LQ"] + [n for n,_ in CANDS] + ["GT"]
rows=[]
for tag, lq, gt in crops:
    imgs=[(lq,"LQ")]
    for name, ck in CANDS:
        set_ckpt(ck)
        imgs.append((infer(lq), name))
    imgs.append((gt,"GT"))
    r=[]
    for im, nm in imgs:
        fn=f"{OUT}/{tag}_{nm}.png"; sv(im, fn); r.append((fn,nm))
    rows.append((tag,r))
    print(f"  {tag} 完成")

W=512; pad,lh=4,20
cv=Image.new("RGB",(len(cols)*W+(len(cols)+1)*pad, len(rows)*(W+lh+pad)+pad),(18,18,18))
d=ImageDraw.Draw(cv)
for ri,(tag,r) in enumerate(rows):
    y=pad+ri*(W+lh+pad)
    for ci,(fn,nm) in enumerate(r):
        x=pad+ci*(W+pad)
        cv.paste(Image.open(fn).convert("RGB").resize((W,W)),(x,y+lh))
        d.text((x+4,y+4),f"{tag} {nm}",fill=(245,245,245))
p=f"{ROOT}/visualizations/iqa_sweep_montage.png"
cv.save(p); print(f"已保存 {p} ({cv.width}x{cv.height})")
