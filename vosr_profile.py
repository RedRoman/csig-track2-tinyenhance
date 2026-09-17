"""给 VOSR 做 TinySR 式特化的可行性剖析。

TinySR = SD3 剪半 + 换 TAESD + 任务特化训练。对 VOSR 做同样三件事：
  1. SD2 VAE encoder (19.57ms) -> 原版 TAESD（4通道，SD2.1 latent 即插即用）
  2. DiT 28 层 -> 12/14 层
  3. 输出端已有 LightDecoder
另外 layer_dinov2b_list=[8] 说明只用了 DINOv2 第 8 层输出，后面 3 个 block 可直接砍掉。
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/VOSR"); os.chdir("/cache/zihaohe/VOSR")
os.environ["TORCH_HOME"] = "/cache/zihaohe/VOSR/preset/ckpts/torch_cache"
import torch, torch.nn.functional as F, numpy as np
from torchvision.transforms import Normalize
from safetensors.torch import load_file
DEV, DT = torch.device("cuda"), torch.float16
CK = "/cache/zihaohe/VOSR_weights/VOSR_0.5B_os"
A = json.load(open(os.path.join(CK, "args.json")))

def bench(fn, n=60, w=15):
    with torch.no_grad():
        for _ in range(w): fn()
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts))

from PIL import Image
Image.MAX_IMAGE_PIXELS=None
_im = np.asarray(Image.open("/cache/zihaohe/csig_isr/data/赛题二/验证集/case2_gt.png").convert("RGB"),
                 np.float32)/255.
_t = torch.from_numpy(_im).permute(2,0,1)[None]
_H,_W = _t.shape[-2:]
x = _t[:,:,_H//2-256:_H//2+256,_W//2-256:_W//2+256].contiguous().to(DEV, dtype=DT)
R = {}

# ---- SD2 VAE ----
from diffusers import AutoencoderKL, AutoencoderTiny
vae = AutoencoderKL.from_pretrained("preset/ckpts/stable-diffusion-2-1-base",
                                    subfolder="vae", torch_dtype=DT).to(DEV).eval()
with torch.no_grad(): z4 = vae.encode(x*2-1).latent_dist.mode()
R["SD2 encode"] = bench(lambda: vae.encode(x*2-1).latent_dist.mode())
R["SD2 decode"] = bench(lambda: vae.decode(z4, return_dict=False)[0])

# ---- 原版 TAESD（SD1.x/2.x 4通道） ----
TAE = "/cache/zihaohe/hf/hub/models--madebyollin--taesd/snapshots/614f76814bbe30edbe2e627ace1c2234c81a2c0e"
tae = AutoencoderTiny.from_pretrained(TAE, torch_dtype=DT).to(DEV).eval()
with torch.no_grad(): zt = tae.encode(x*2-1).latents
R["TAESD-orig encode"] = bench(lambda: tae.encode(x*2-1).latents)
R["TAESD-orig decode"] = bench(lambda: tae.decode(zt, return_dict=False)[0])
print(f"latent 形状  SD2 {tuple(z4.shape)}   TAESD-orig {tuple(zt.shape)}", flush=True)

# TAESD 的 latent 与 SD2 的 latent 是否在同一空间（互换解码看得出来）
with torch.no_grad():
    def psnr(a,b): return 10*np.log10(1.0/max(float(((a-b)**2).mean()),1e-12))
    x01 = x.float()
    rt_k = ((vae.decode(z4, return_dict=False)[0]*0.5+0.5).clamp(0,1)).float()
    rt_t = ((tae.decode(zt, return_dict=False)[0]*0.5+0.5).clamp(0,1)).float()
    cross = ((vae.decode(zt, return_dict=False)[0]*0.5+0.5).clamp(0,1)).float()   # TAESD编码->SD2解码
    print(f"[真实图 case2_gt 中心512块] SD2自往返 {psnr(rt_k,x01):.2f}dB  "
          f"TAESD自往返 {psnr(rt_t,x01):.2f}dB  TAESD编码->SD2解码 {psnr(cross,x01):.2f}dB", flush=True)
    print(f"  cross 与 SD2自往返 的相关 "
          f"{float(torch.corrcoef(torch.stack([cross.flatten(), rt_k.flatten()]))[0,1]):.4f}", flush=True)

# ---- LightDecoder ----
from models.light_decoder import LightDecoder
ck = torch.load("preset/ckpts/sd21_lwdecoder.pth", map_location="cpu")
c = ck["config"]
ld = LightDecoder(in_channels=c["in_channels"], out_channels=c["out_channels"],
                  block_out_channels=tuple(c["block_out_channels"]),
                  layers_per_block=c["layers_per_block"])
ld.load_state_dict(ck["model_state_dict"]); ld = ld.to(DEV, dtype=DT).eval()
R["LightDecoder"] = bench(lambda: ld(z4))

# ---- DINOv2 ----
enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", source="github" if False else "local",
                     trust_repo=True) if False else torch.hub.load(
    "/cache/zihaohe/VOSR/preset/ckpts/torch_cache/facebookresearch_dinov2_main",
    "dinov2_vitb14", source="local", pretrained=False)
sd = torch.load("/cache/zihaohe/VOSR/preset/ckpts/torch_cache/checkpoints/dinov2_vitb14_pretrain.pth",
                map_location="cpu")
enc.load_state_dict(sd, strict=False)
if hasattr(enc, "head"): del enc.head
enc = enc.to(DEV, dtype=DT).eval()
dsz = A.get("dinov2_size", 448)
xd = F.interpolate(x.float(), dsz, mode="bicubic").clip(0,1)
xd = Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(xd).to(DT)
R[f"DINOv2-B @{dsz} 全12层"] = bench(lambda: enc.forward_features(xd)["x_norm_patchtokens"])
zf = enc.forward_features(xd)["x_norm_patchtokens"]
nblk = len(enc.blocks)
enc.blocks = torch.nn.ModuleList(list(enc.blocks)[:9])   # 只用到第8层，砍掉后面
R[f"DINOv2-B @{dsz} 截到9层"] = bench(lambda: enc.forward_features(xd)["x_norm_patchtokens"])
xd224 = Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(
    F.interpolate(x.float(), 224, mode="bicubic").clip(0,1)).to(DT)
R["DINOv2-B @224 截到9层"] = bench(lambda: enc.forward_features(xd224)["x_norm_patchtokens"])

# ---- DiT ----
from models.lightningdit import LightningDiT
def mk(depth):
    m = LightningDiT(input_size=A["resolution"]//8, patch_size=A["patch_size"],
                     in_channels=8, out_channels=4, hidden_size=A["dim"], depth=depth,
                     num_heads=A["num_heads"], mlp_ratio=A["mlp_ratio"], z_dims=A["enc_dim"],
                     encdim_ratio=A["encdim_ratio"], auxiliary_time_cond=False,
                     use_qknorm=A["use_qknorm"], use_swiglu=A["use_swiglu"],
                     use_rope=A["use_rope"], use_rmsnorm=A["use_rmsnorm"],
                     wo_shift=A["wo_shift"], num_fused_layers=len(A["layer_dinov2b_list"]))
    return m.to(DEV, dtype=DT).eval()
m28 = mk(28)
w = load_file(os.path.join(CK, "checkpoints", "ema_model.safetensors"))
r = m28.load_state_dict({k: v.to(DT) for k, v in w.items()}, strict=False)
print(f"DiT 参数 {sum(p.numel() for p in m28.parameters())/1e6:.1f}M  "
      f"缺失 {len(r.missing_keys)}  多余 {len(r.unexpected_keys)}", flush=True)
inp = torch.cat([z4, z4], 1); tt = torch.zeros(1, device=DEV, dtype=DT)
def run(m):
    with torch.autocast("cuda", dtype=torch.float16):
        return m(inp, tt, None, [zf])
_ = run(m28)
for d in (28, 20, 14, 12):
    md = m28 if d == 28 else mk(d)
    R[f"VOSR DiT {d}层"] = bench(lambda mm=md: run(mm))
    if d != 28: del md; torch.cuda.empty_cache()

print(f"\n{'模块':<28}{'中位 ms':>12}")
for k, v in R.items(): print(f"{k:<28}{v:>12.2f}")

d9 = R[f"DINOv2-B @{dsz} 截到9层"]
print("\n=== 组合 ===")
orig = R["SD2 encode"] + R[f"DINOv2-B @{dsz} 全12层"] + R["VOSR DiT 28层"] + R["LightDecoder"]
print(f"VOSR 原样                     {orig:6.2f} ms  (加速比 {100.15/orig:.2f}×)")
for d in (28, 20, 14, 12):
    t = R["TAESD-orig encode"] + d9 + R[f"VOSR DiT {d}层"] + R["LightDecoder"]
    print(f"TAESD编码+DINO9层+DiT{d:>2}层+LightDec  {t:6.2f} ms  (加速比 {100.15/t:.2f}×)")
print(f"对照 TinySR 现役               12.75 ms  (加速比 {100.15/12.75:.2f}×)")
json.dump(R, open("/cache/zihaohe/csig_isr/eval_out/vosr_profile.json","w"), indent=2, ensure_ascii=False)
