"""导出 v2：对比 freeze / 不 freeze，选文件更小且速度不劣的版本。

v1 用 freeze(optimize_numerics=False)，权重全变成图常量（parameters() 为空），
文件 686MB 且无法压缩，可能是主办方环境加载卡住的原因。
"""
import os, sys, json, gc
import torch, torch.nn as nn
import numpy as np
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
OUT=f"{ROOT}/submit/model_dir"

class SubmitModel(nn.Module):
    def __init__(s, vae, tr, ts, pooled):
        super().__init__(); s.vae, s.tr = vae, tr
        s.register_buffer("ts", ts, persistent=False)
        s.register_buffer("pooled", pooled, persistent=False)
    def forward(s, x_m11):
        sf=s.vae.config.scaling_factor
        z=s.vae.encode(x_m11).latents*sf
        p=s.tr(hidden_states=z, timestep=s.ts, pooled_projections=s.pooled, return_dict=False)[0]
        return s.vae.decode((z-p)/sf, return_dict=False)[0].clamp(-1,1)

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
tr=tr.merge_and_unload()
model=SubmitModel(vae,tr,ts,pooled).to(dev).half().eval()
print(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f} M "
      f"(fp16 理论 {sum(p.numel() for p in model.parameters())*2/1e6:.1f} MB)")

a=np.asarray(Image.open(f"{ROOT}/data/赛题二/测试集/case1.jpg").convert("RGB"),dtype=np.float32)/255
x=((torch.from_numpy(a).permute(2,0,1)[None][:,:,1000:1512,1000:1512])*2-1).to(dev,dtype=dt)
with torch.no_grad(): ref=model(x).float().clone()

def bench(fn,n=60,warm=25):
    with torch.no_grad():
        for _ in range(warm): fn(x)
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): fn(x)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

with torch.no_grad():
    traced=torch.jit.trace(model,x,check_trace=False).eval()

results={}
for tag,mk,fn in [("trace_only","tinysr_x1_trace.pt",lambda m:m),
                  ("frozen","tinysr_x1_frozen.pt",
                   lambda m: torch.jit.freeze(m,optimize_numerics=False))]:
    try:
        mm=fn(traced)
    except Exception as ex:
        print(f"{tag} 失败: {str(ex)[:60]}"); continue
    p=os.path.join(OUT,mk); mm.save(p)
    sz=os.path.getsize(p)/1e6
    del mm; gc.collect(); torch.cuda.empty_cache()
    L=torch.jit.load(p,map_location=dev); L.eval()
    with torch.no_grad():
        for _ in range(25): L(x)
    torch.cuda.synchronize()
    with torch.no_grad(): o=L(x).float()
    rel=(o-ref).abs().mean().item()/max(ref.abs().mean().item(),1e-8)
    lat=min(bench(L) for _ in range(5))
    npar=sum(q.numel() for q in L.parameters())
    print(f"{tag:<12} {sz:>7.1f} MB  时延 {lat:>6.2f} ms  差异 {rel*100:.4f}%  "
          f"parameters {npar/1e6:.2f}M")
    results[tag]={"size_MB":sz,"lat":lat,"rel":rel,"params_M":npar/1e6}
    del L; gc.collect(); torch.cuda.empty_cache()

json.dump(results,open(f"{ROOT}/eval_out/export_v2.json","w"),indent=1)
