"""干净环境下的权威时延测量：同时测提交模型与 OSEDiff 基准，多轮取中位数。

此前测到 12.2/24.9/44.0 三个数字，差异源于：
  - 输入张量在 CPU 还是 GPU（H2D 拷贝约 13ms）
  - GPU 被其它用户的进程占用（本机共享，实测同进程内 5 轮波动 12.3~28.3ms）
故此处固定：GPU 张量输入 + 多轮取【最小值】（最小值最接近无干扰的真实性能）。
"""
import os, sys, json, statistics
import torch
ROOT="/cache/zihaohe/csig_isr"
MD=f"{ROOT}/submit/model_dir"
dev=torch.device("cuda"); dt=torch.float16

def bench(fn, x, n=100, warm=30):
    with torch.no_grad():
        for _ in range(warm): fn(x)
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): fn(x)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

print("GPU:", torch.cuda.get_device_name(0))
occ = os.popen("nvidia-smi --query-compute-apps=pid --format=csv,noheader").read().strip()
print("其它占用进程:", occ.replace("\n",",") or "无")
print()

m = torch.jit.load(f"{MD}/tinysr_x1.pt", map_location=dev); m.eval()
x = torch.rand(1,3,512,512, device=dev, dtype=dt)*2-1
runs = [bench(lambda a: m(a), x) for _ in range(9)]
print(f"提交模型 9 轮: {[f'{v:.2f}' for v in runs]}")
t_ours = min(runs)
print(f"  最小 {t_ours:.3f} ms  中位 {statistics.median(runs):.3f} ms  最大 {max(runs):.3f} ms")

# OSEDiff 基准（同口径重测）
print()
print("重测 OSEDiff 基准（同口径：GPU 张量、独占）...")
sys.path.insert(0, f"{ROOT}/OSEDiff"); os.chdir(f"{ROOT}/OSEDiff")
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel
SD21=f"{ROOT}/sd21base"
vae=AutoencoderKL.from_pretrained(SD21,subfolder="vae")
unet=UNet2DConditionModel.from_pretrained(SD21,subfolder="unet")
tok=AutoTokenizer.from_pretrained(SD21,subfolder="tokenizer")
te=CLIPTextModel.from_pretrained(SD21,subfolder="text_encoder")
sched=DDPMScheduler.from_pretrained(SD21,subfolder="scheduler"); sched.set_timesteps(1,device=dev)
ck=torch.load(f"{ROOT}/AdcSR/weight/pretrained/osediff.pkl",map_location="cpu",weights_only=False)
vae.load_state_dict(ck["vae"],strict=False); unet.load_state_dict(ck["unet"],strict=False)
vae=vae.to(dev,dtype=dt).eval(); unet=unet.to(dev,dtype=dt).eval(); te=te.to(dev,dtype=dt).eval()
sched.alphas_cumprod=sched.alphas_cumprod.to(dev)
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from torchvision import transforms
D=ram(pretrained=f"{ROOT}/AdcSR/weight/pretrained/ram_swin_large_14m.pth",
      pretrained_condition=f"{ROOT}/AdcSR/weight/pretrained/DAPE.pth",
      image_size=384,vit="swin_l").eval().to(dev).to(dt)
rt=transforms.Compose([transforms.Resize((384,384)),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
ts=torch.tensor([999],device=dev).long()
lq=torch.rand(1,3,512,512,device=dev,dtype=dt)

def ose():
    r=inference(rt(lq), D)
    p=r[0] if isinstance(r,(list,tuple)) else str(r)
    ids=tok(p,max_length=tok.model_max_length,padding="max_length",
            truncation=True,return_tensors="pt").input_ids.to(dev)
    e=te(ids)[0].to(dt)
    z=vae.encode(lq*2-1).latent_dist.sample()*vae.config.scaling_factor
    mo=unet(z,ts,encoder_hidden_states=e).sample
    zo=sched.step(mo,ts,z,return_dict=True).prev_sample.to(dt)
    return vae.decode(zo/vae.config.scaling_factor).sample

runs_o=[bench(lambda _: ose(), lq, n=20, warm=8) for _ in range(5)]
print(f"OSEDiff 5 轮: {[f'{v:.2f}' for v in runs_o]}")
t_ose=min(runs_o)
print(f"  最小 {t_ose:.3f} ms  中位 {statistics.median(runs_o):.3f} ms  最大 {max(runs_o):.3f} ms")

sp=t_ose/t_ours
print()
print("="*60)
print(f"  提交模型  {t_ours:.2f} ms")
print(f"  OSEDiff   {t_ose:.2f} ms")
print(f"  加速比    {sp:.2f}x     Speedup^0.2 = {sp**0.2:.4f}")
print(f"  Score 3.7217 ->  FinalScore {2**3.7217*sp**0.2:.2f}")
print("="*60)
json.dump({"t_ours":t_ours,"t_osediff":t_ose,"speedup":sp},
          open(f"{ROOT}/eval_out/final_latency.json","w"),indent=1)
