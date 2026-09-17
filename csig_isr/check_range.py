"""验证提交模型的值域转换是否正确。

训练/出图路径（infer_4k.py core）：
    输入 x01 in [0,1]  ->  x = x01*2-1  ->  VAE.encode(x)  ->  ...
    ->  VAE.decode  ->  (img*0.5+0.5).clamp(0,1)   输出 [0,1]

提交模型要求 infer 输入输出均 [-1,1]。正确写法应为：
    x_m11 in [-1,1]  ->  VAE.encode(x_m11)  ->  ...  ->  decode  ->  输出直接是 [-1,1]
即 encode 前不需要再转换（因为训练时 encode 的输入本就是 [-1,1] 量纲）。
"""
import os, sys
import torch, numpy as np
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0,TS); os.chdir(TS)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
dev,dt=torch.device("cuda"),torch.float16

tr=TinySD3Transformer2DModel.from_pretrained(
    f"{TS}/checkpoint/tinybackbone/prune-12-merge-tinysr",subfolder="transformer",
    torch_dtype=dt,low_cpu_mem_usage=False,ignore_mismatched_sizes=True)
vae=AutoencoderTiny.from_pretrained(f"{TS}/checkpoint/vae/separable",torch_dtype=dt)
lc=LoraConfig(r=64,lora_alpha=64,init_lora_weights="gaussian",
    target_modules=["to_k","to_q","to_v","to_out.0","proj","linear","linear_1","linear_2","net.2"])
tr.add_adapter(lc); tr.enable_adapters()
load_lora_state_dict(StableDiffusion3Pipeline.lora_state_dict(
    f"{TS}/checkpoint/tinysr",weight_name="transformer.safetensors"),tr)
vae=vae.to(dev,dtype=dt).eval(); tr=tr.to(dev,dtype=dt).eval()
ts=torch.tensor([1000.0],device=dev,dtype=dt)
pooled=torch.load(f"{TS}/dataset/default/pool_embeds.pt",map_location=dev).to(dtype=dt)
with torch.no_grad():
    w=torch.rand(1,3,512,512,device=dev,dtype=dt)*2-1
    zz=vae.encode(w).latents*vae.config.scaling_factor
    _=tr(hidden_states=zz,timestep=ts,pooled_projections=pooled,return_dict=False)
sd=torch.load(f"{ROOT}/train_out/tinysr8_fid/best.pth",map_location="cpu",weights_only=False)
cur=tr.state_dict()
for k,v in sd["lora"].items():
    if k in cur and cur[k].shape==v.shape: cur[k].copy_(v.to(cur[k].dtype))

# 用真实图片
a=np.asarray(Image.open(f"{ROOT}/data/赛题二/测试集/case1.jpg").convert("RGB"),dtype=np.float32)/255
t=torch.from_numpy(a).permute(2,0,1)[None][:,:,1000:1512,1000:1512]
x01=t.to(dev,dtype=dt)              # [0,1]
x_m11=x01*2-1                        # [-1,1]

@torch.no_grad()
def pipeline(enc_input):
    sf=vae.config.scaling_factor
    z=vae.encode(enc_input).latents*sf
    p=tr(hidden_states=z,timestep=ts,pooled_projections=pooled,return_dict=False)[0]
    return vae.decode((z-p)/sf,return_dict=False)[0]

print("输入统计: x01 range [%.3f, %.3f]  x_m11 range [%.3f, %.3f]"%(
    x01.min(),x01.max(),x_m11.min(),x_m11.max()))
print()
with torch.no_grad():
    # 训练/出图的正确路径：encode 收 [-1,1]
    out_a = pipeline(x_m11)
    print("A. encode(x_m11)  [训练路径]      输出 range [%.4f, %.4f]  均值 %.4f"%(
        out_a.min(),out_a.max(),out_a.mean()))
    # 我在 SubmitModel 里的写法（同 A，因为直接传 x_m11）
    # 错误对照：encode 收 [0,1]
    out_b = pipeline(x01)
    print("B. encode(x01)    [错误对照]      输出 range [%.4f, %.4f]  均值 %.4f"%(
        out_b.min(),out_b.max(),out_b.mean()))
print()
print("导出脚本报的 [-0.100, 0.181] 是用【随机噪声】测的，不是真实图片。")
print("随机噪声经 VAE 后本就趋于均值，值域窄属正常，不代表模型有问题。")
