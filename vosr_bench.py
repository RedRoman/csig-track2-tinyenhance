"""VOSR 0.5B 单步在 512x512 的组件级时延（同机同协议，对标我们的 10.67 ms）。

一票否决判据：赛题时延测的是 512 单次前向。若总时延远超 20 ms，
按 p≈0.2 需要 Score +0.24 才回本，而理论空间只有 +0.042 —— 直接否掉。
"""
import os, sys, time, types, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/VOSR")
os.chdir("/cache/zihaohe/VOSR")
os.environ.setdefault("TORCH_HOME", "/cache/zihaohe/VOSR_weights/torch_cache")
import torch, torch.nn.functional as F, json
from torchvision.transforms import Normalize
from safetensors.torch import load_file

DEV = torch.device("cuda")
CKPT = "/cache/zihaohe/VOSR_weights/VOSR_0.5B_os"
args = json.load(open(os.path.join(CKPT, "args.json")))
print("args 关键项:", {k: args.get(k) for k in
      ("ae_type","resolution","patch_size","enc_type","dinov2_size",
       "hidden_size","depth","num_heads") if k in args})

def bench(fn, n=30, w=10):
    with torch.no_grad():
        for _ in range(w): fn()
        torch.cuda.synchronize(); best = 1e9
        for _ in range(4):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(n): fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter()-t0)/n*1000)
    return best

# VAE
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("preset/ckpts/stable-diffusion-2-1-base",
                                    subfolder="vae").to(DEV).eval().half()
# LightDecoder
from inference_vosr_onestep import LightDecoder
ck = torch.load("preset/ckpts/sd21_lwdecoder.pth", map_location="cpu")
c = ck["config"]
ld = LightDecoder(in_channels=c["in_channels"], out_channels=c["out_channels"],
                  block_out_channels=tuple(c["block_out_channels"]),
                  layers_per_block=c["layers_per_block"])
ld.load_state_dict(ck["model_state_dict"]); ld = ld.to(DEV).eval().half()
# DINOv2
enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
del enc.head; enc = enc.to(DEV).eval().half()
# DiT
from models.lightningdit import LightningDiT
base_c = 4
m = LightningDiT(input_size=args["resolution"]//8, patch_size=args["patch_size"],
                 in_channels=2*base_c, **{k: args[k] for k in
                 ("hidden_size","depth","num_heads") if k in args})
sd = load_file(os.path.join(CKPT, "checkpoints", "ema_model.safetensors"))
miss = m.load_state_dict(sd, strict=False)
m = m.to(DEV).eval().half()
print(f"DiT 参数 {sum(p.numel() for p in m.parameters())/1e6:.1f} M  "
      f"缺失键 {len(miss.missing_keys)}")

x = torch.rand(1,3,512,512, device=DEV, dtype=torch.half)
dsz = args.get("dinov2_size", 224)
xd = F.interpolate(x.float(), dsz, mode="bicubic").clip(0,1)
xd = Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(xd).half()

t_enc = bench(lambda: vae.encode(x*2-1))
lat = vae.encode(x*2-1).latent_dist.sample()
t_dino = bench(lambda: enc.forward_features(xd)["x_norm_patchtokens"])
zf = enc.forward_features(xd)["x_norm_patchtokens"]
inp = torch.cat([lat, lat], 1)
tt = torch.zeros(1, device=DEV, dtype=torch.half)
try:
    t_dit = bench(lambda: m(inp, tt, tt, zf))
except Exception as e:
    print("DiT 单测失败:", type(e).__name__, str(e)[:100]); t_dit = float("nan")
t_dec = bench(lambda: ld(lat))

print()
print(f"  VAE encode      {t_enc:7.2f} ms")
print(f"  DINOv2 编码器    {t_dino:7.2f} ms")
print(f"  LightningDiT    {t_dit:7.2f} ms")
print(f"  LightDecoder    {t_dec:7.2f} ms")
print(f"  合计            {t_enc+t_dino+t_dit+t_dec:7.2f} ms")
print(f"  我们 TinySR      {10.67:7.2f} ms")
