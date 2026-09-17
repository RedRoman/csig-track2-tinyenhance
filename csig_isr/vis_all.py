"""四方视觉对比：LQ | Samsung(训练后) | TinySR+wavelet | GT
同时给出每张 crop 的逐指标数字，便于对照肉眼观感。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
VAL = f"{ROOT}/data/赛题二/验证集"
OUT = f"{ROOT}/visualizations/four_way"
os.makedirs(OUT, exist_ok=True)
dev, dt = torch.device("cuda"), torch.float16
PAIRS = [("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
         ("case3_lq.jpg","case3_gt.jpg",303)]


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1)[None]


# ---------- Samsung ----------
sys.path.insert(0, f"{ROOT}/NTIRE2026")
from diffusers import UNet2DConditionModel
from models.team11_SamsungAKCamera.model import create_srnet

class ResSRNet(nn.Module):
    def __init__(s, net, refine, decoder):
        super().__init__(); s.net, s.refine, s.decoder = net, refine, decoder
        s.zero_conv = nn.Conv2d(3,3,1)
    def forward(s, x):
        return (x + s.zero_conv(s.decoder(s.refine(s.net(x))) - 0.5)).clamp(0,1)

cfg = UNet2DConditionModel.load_config(f"{ROOT}/sd21base", subfolder="unet")
m = create_srnet(UNet2DConditionModel.from_config(cfg),
                 checkpoint_path=f"{ROOT}/NTIRE2026/model_zoo/team11_SamsungAICamera/team11_mobile_samsung/weights.pth")
m.net.body[1].conv_in = nn.Conv2d(192,160,3,padding=1); m.net.body[0] = nn.PixelUnshuffle(8)
samsung = ResSRNet(m.net, m.refine, m.decoder)
sd = torch.load(f"{ROOT}/train_out/run1/best.pth", map_location="cpu", weights_only=False)
samsung.net.load_state_dict(sd["net"]); samsung.refine.load_state_dict(sd["refine"])
samsung.decoder.load_state_dict(sd["decoder"]); samsung.zero_conv.load_state_dict(sd["zero_conv"])
samsung = samsung.to(dev).half().eval()
print(f"Samsung 载入完成 (step {sd.get('step')})")

# ---------- TinySR ----------
TS = f"{ROOT}/TinySR"
# NTIRE2026 与 TinySR 都有顶层 models 包，需清缓存并把 TinySR 置于路径最前
for _k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
    del sys.modules[_k]
sys.path.insert(0, TS)
_cwd = os.getcwd(); os.chdir(TS)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from utils.wavelet_color_fix import wavelet_color_fix
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
from torchvision.transforms import ToPILImage, ToTensor

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
os.chdir(_cwd)
print("TinySR 载入完成")

@torch.no_grad()
def tinysr(lq01):
    x = lq01.to(dev,dtype=dt)*2-1
    z = vae.encode(x).latents*vae.config.scaling_factor
    p = tr(hidden_states=z, timestep=ts, pooled_projections=pooled, return_dict=False)[0]
    img = vae.decode((z-p)/vae.config.scaling_factor, return_dict=False)[0]
    sr = (img.float()*0.5+0.5).clamp(0,1)
    fixed = wavelet_color_fix(ToPILImage()(sr[0].cpu()), ToPILImage()(lq01[0].cpu()))
    return ToTensor()(fixed)[None].to(dev)

# ---------- 取 crop ----------
crops=[]
for lqn, gtn, seed in PAIRS:
    lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
    if lq_f.shape != gt_f.shape: continue
    rng = np.random.RandomState(seed); _,_,H,W = gt_f.shape
    got, tries = 0, 0
    while got < 2 and tries < 40:
        tries += 1
        y,x = rng.randint(0,H-512), rng.randint(0,W-512)
        gt = gt_f[:,:,y:y+512,x:x+512]
        if gt.std().item() < 0.06: continue
        got += 1
        crops.append((f"{lqn[:5]}_{got}", lq_f[:,:,y:y+512,x:x+512].clone(), gt.clone()))
print(f"crop {len(crops)}")

import pyiqa
mp = pyiqa.create_metric("psnr", device=dev, test_y_channel=True, color_space="ycbcr")
ml = pyiqa.create_metric("lpips", device=dev)
mc = pyiqa.create_metric("clipiqa", device=dev)

def sv(t,p):
    a=(t[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
    Image.fromarray(a).save(p)

print(f"\n{'crop':<10}{'方法':<18}{'PSNR':>8}{'LPIPS':>9}{'CLIPIQA':>9}")
print("-"*56)
tiles=[]
for tag, lq, gt in crops:
    gtd = gt.to(dev); lqd = lq.to(dev)
    with torch.no_grad():
        s_out = samsung(lqd.half()).float()
    t_out = tinysr(lq)
    row=[]
    for name, im in [("LQ 输入", lqd), ("Samsung 训练后", s_out), ("TinySR+wavelet", t_out), ("GT 参考", gtd)]:
        if name == "GT 参考":
            p, l, c = 99.0, 0.0, mc(im).item()
        else:
            p, l, c = mp(im,gtd).item(), ml(im,gtd).item(), mc(im).item()
        print(f"{tag:<10}{name:<18}{p:>8.2f}{l:>9.4f}{c:>9.4f}")
        fn=f"{OUT}/{tag}_{name.split()[0]}.png"; sv(im, fn)
        row.append((fn, f"{name}  PSNR{p:.1f} LPIPS{l:.3f} CLIPIQA{c:.3f}" if p<99 else f"{name}  CLIPIQA{c:.3f}"))
    tiles.append((tag,row))
    print()

# 拼图
W=512; pad, lh = 6, 24
canvas = Image.new("RGB",(4*W+5*pad, len(tiles)*(W+lh+pad)+pad),(20,20,20))
d = ImageDraw.Draw(canvas)
for ri,(tag,row) in enumerate(tiles):
    y = pad+ri*(W+lh+pad)
    for ci,(fn,lab) in enumerate(row):
        x = pad+ci*(W+pad)
        canvas.paste(Image.open(fn).convert("RGB").resize((W,W)),(x,y+lh))
        d.text((x+4,y+5), lab, fill=(240,240,240))
p=f"{ROOT}/visualizations/four_way_montage.png"
canvas.save(p)
print(f"已保存 {p}  ({canvas.width}x{canvas.height})")
