"""只生成 Samsung 训练后的输出图（与 dump_cmp.py 用相同的 crop 选取规则）。"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
VAL = f"{ROOT}/data/赛题二/验证集"
OUT = f"{ROOT}/visualizations/x1_compare"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
sys.path.insert(0, f"{ROOT}/NTIRE2026")
from diffusers import UNet2DConditionModel
from models.team11_SamsungAKCamera.model import create_srnet

PAIRS = [("case1_lq.jpg","case1_gt.png",101),("case2_lq.jpg","case2_gt.png",202),
         ("case3_lq.jpg","case3_gt.jpg",303)]

def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1)[None]

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
model = ResSRNet(m.net, m.refine, m.decoder)
sd = torch.load(f"{ROOT}/train_out/run1/best.pth", map_location="cpu", weights_only=False)
model.net.load_state_dict(sd["net"]); model.refine.load_state_dict(sd["refine"])
model.decoder.load_state_dict(sd["decoder"]); model.zero_conv.load_state_dict(sd["zero_conv"])
model = model.to(dev).half().eval()
print(f"Samsung 载入 (step {sd.get('step')})")

import pyiqa
mp = pyiqa.create_metric("psnr", device=dev, test_y_channel=True, color_space="ycbcr")
ml = pyiqa.create_metric("lpips", device=dev)
mc = pyiqa.create_metric("clipiqa", device=dev)

def sv(t,p):
    a=(t[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
    Image.fromarray(a).save(p)

print(f"\n{'crop':<10}{'方法':<16}{'PSNR':>8}{'LPIPS':>9}{'CLIPIQA':>9}")
print("-"*54)
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
        tag=f"{lqn[:5]}_{got}"
        lq = lq_f[:,:,y:y+512,x:x+512].to(dev)
        gtd = gt.to(dev)
        with torch.no_grad():
            out = model(lq.half()).float()
        sv(out, f"{OUT}/{tag}_4_Samsung.png")
        print(f"{tag:<10}{'Samsung':<16}{mp(out,gtd).item():>8.2f}{ml(out,gtd).item():>9.4f}{mc(out).item():>9.4f}")
        print(f"{tag:<10}{'LQ':<16}{mp(lq,gtd).item():>8.2f}{ml(lq,gtd).item():>9.4f}{mc(lq).item():>9.4f}")
print(f"\n已导出到 {OUT}")
