"""基于瓶颈诊断的模块设计与验证。

诊断结论：
  1. decoder 是硬瓶颈（上限 LPIPS 0.5167，当前已达 0.5003 反超）
  2. 误差 81% 在中高频（中 39.1% + 高 41.9%）
  3. 低频误差仅 19.1%（保真约束已生效）

=> 设计原则：绕开 decoder 的低通特性，让高频走独立路径

三个候选模块：
  A. 像素域高频旁路（HF Bypass）
     decoder 输出 + 一个直接在像素域预测高频残差的轻量分支
     理由：decoder 的 8x 上采样天然低通，高频不该经它
  B. 频域自适应融合（Freq-Adaptive Fusion）
     把 LQ 的低频 + decoder 的中频 + 旁路的高频按频段加权合成
  C. 双尺度 latent（Dual-Scale）
     除 64x64 latent 外，额外在 128x128 上预测一路细节
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0,TS); os.chdir(TS)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
dev,dt=torch.device("cuda"),torch.float16

# ---------------- 模块 A: 像素域高频旁路 ----------------
class HFBypass(nn.Module):
    """在像素域直接预测高频残差，绕开 decoder 的低通瓶颈。

    输入 LQ 512x512（全分辨率，不经 VAE 压缩），
    用轻量卷积预测应补的高频，加到 decoder 输出上。
    zero-init 输出层 => 起点等价于原模型。
    """
    def __init__(self, ch=32, n_block=4):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
                          nn.Conv2d(ch, ch, 3, padding=1))
            for _ in range(n_block)])
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)
        self.act = nn.SiLU()
    def forward(self, lq, dec_out):
        f = self.act(self.head(lq))
        for b in self.body:
            f = f + b(f)
        return dec_out + self.tail(f)

# ---------------- 模块 B: 频域自适应融合 ----------------
class FreqFusion(nn.Module):
    """按频段融合：低频取 LQ（实测无损），中高频取 decoder 输出，
    融合权重由一个极小的网络按内容自适应预测。"""
    def __init__(self, ch=16):
        super().__init__()
        self.g = nn.Sequential(
            nn.Conv2d(6, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, 3, 3, padding=1), nn.Sigmoid())
        # 初始化为全通过 decoder（等价原模型）
        nn.init.zeros_(self.g[-2].weight); nn.init.constant_(self.g[-2].bias, 4.0)
    def forward(self, lq, dec_out, k=16):
        lo_lq = F.interpolate(F.avg_pool2d(lq, k), size=lq.shape[-2:],
                              mode="bilinear", align_corners=False)
        lo_de = F.interpolate(F.avg_pool2d(dec_out, k), size=lq.shape[-2:],
                              mode="bilinear", align_corners=False)
        hi_de = dec_out - lo_de
        w = self.g(torch.cat([lq, dec_out], 1))     # 每像素每通道的融合权重
        lo = w * lo_de + (1 - w) * lo_lq
        return lo + hi_de

def count(m): return sum(p.numel() for p in m.parameters())
def bench(f, n=100, warm=30):
    with torch.no_grad():
        for _ in range(warm): f()
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): f()
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

lq=torch.rand(1,3,512,512,device=dev,dtype=dt)
dec=torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
print("="*68)
print("候选模块的参数量与时延代价")
print("="*68)
print(f"{'模块':<28}{'参数':>10}{'时延ms':>10}{'占比':>9}")
print("-"*68)
BASE=12.20
for tag, m in [("A. HF旁路 ch32 x4", HFBypass(32,4)),
               ("A. HF旁路 ch48 x6", HFBypass(48,6)),
               ("A. HF旁路 ch16 x3", HFBypass(16,3)),
               ("B. 频域融合 ch16", FreqFusion(16))]:
    m=m.to(dev).half().eval()
    t=min(bench(lambda: m(lq,dec)) for _ in range(3))
    print(f"{tag:<28}{count(m)/1e3:>9.1f}K{t:>10.3f}{t/BASE*100:>8.1f}%")
print()
print(f"基线时延 {BASE} ms，加速比 8.19x")
print(f"按公式，时延每 +1ms 扣 {((3.8783/6*100)*(99.96/BASE)**0.0708-(3.8783/6*100)*(99.96/(BASE+1))**0.0708)*0.9937:.3f} 分")
