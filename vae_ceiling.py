"""保真度地板：GT 只过一遍 VAE 编解码，完全不经过 DiT。

若 TAESD 往返的 DISTS 已接近我们模型实测的 0.264，
说明瓶颈在 VAE 而不在主干——那么换 DiT / 加层 / 换基座全是使错了劲。
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random
from PIL import Image
from settings import VAL, TS
from metrics_utils import create_metrics
from tinysr_runtime import build_model
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
GTS = ["case1_gt.png","case2_gt.png","case3_gt.jpg"]
LQS = ["case1_lq.jpg","case2_lq.jpg","case3_lq.jpg"]
M = create_metrics(DEV)

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]

def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)

@torch.no_grad()
def rt_tiled(fn, x, tile=512, ov=128):
    _,_,H,W = x.shape; st = tile-ov
    ys = list(range(0,max(H-tile,0)+1,st)) or [0]
    xs = list(range(0,max(W-tile,0)+1,st)) or [0]
    if ys[-1]+tile<H: ys.append(max(H-tile,0))
    if xs[-1]+tile<W: xs.append(max(W-tile,0))
    t = torch.linspace(-1,1,tile,device=DEV); g = torch.exp(-(t**2)/(2*0.45**2))
    wm = (g[:,None]*g[None,:])[None,None].float()
    out = torch.zeros(1,3,H,W,device=DEV); acc = torch.zeros(1,1,H,W,device=DEV)
    for y in ys:
        for xx in xs:
            p = fn(x[:,:,y:y+tile,xx:xx+tile].to(DEV,dtype=DT)).float()
            out[:,:,y:y+tile,xx:xx+tile]+=p*wm; acc[:,:,y:y+tile,xx:xx+tile]+=wm
    return (out/acc.clamp_min(1e-8)).clamp(0,1)

def mk_tae(vae):
    sf = vae.config.scaling_factor
    def f(x01):
        z = vae.encode(x01*2-1).latents * sf
        img = vae.decode(z/sf, return_dict=False)[0]
        return (img*0.5+0.5).clamp(0,1)
    return f

VAES = []
# 1) 原装 TAESD（TinySR 出厂）
sys.path.insert(0, f"{R}/v3/code")
from models.vae.autoencoder_tiny import AutoencoderTiny
tae0 = AutoencoderTiny.from_pretrained(os.path.join(TS,"checkpoint","vae","separable"),
                                       torch_dtype=DT).to(DEV, dtype=DT).eval()
VAES.append(("TAESD-原装", mk_tae(tae0)))
# 2) run12 微调过的 decoder
m12,_ = build_model(f"{R}/train_out/run12_iqa30/best.pth", device=DEV, dtype=DT,
                    merge_lora=False, alpha_mode="fixed", fixed_alpha=1.0)
m12.eval(); VAES.append(("TAESD-run12微调", mk_tae(m12.vae)))
# 3) SD2 VAE（VOSR / OSEDiff 用的）
from diffusers import AutoencoderKL
kl = AutoencoderKL.from_pretrained(f"{R}/sd21base", subfolder="vae",
                                   torch_dtype=DT).to(DEV, dtype=DT).eval()
def kl_rt(x01):
    z = kl.encode(x01*2-1).latent_dist.mode()
    return (kl.decode(z, return_dict=False)[0]*0.5+0.5).clamp(0,1)
VAES.append(("SD2-VAE", kl_rt))

rows = {}
gt_t = [load(os.path.join(VAL,g)).to(DEV) for g in GTS]
for src_tag, names in (("GT", GTS), ("LQ", LQS)):
    imgs = [load(os.path.join(VAL,n)) for n in names]
    for vtag, fn in VAES:
        seed_all(); L,D,P = [],[],[]
        for x, gt in zip(imgs, gt_t):
            y = rt_tiled(fn, x)
            L.append(float(M["lpips"](y,gt))); D.append(float(M["dists"](y,gt)))
            P.append(10*np.log10(1.0/max(float(((y-gt)**2).mean()),1e-12)))
            del y; torch.cuda.empty_cache()
        rows[f"{src_tag} 过 {vtag}"] = dict(lpips=float(np.mean(L)), dists=float(np.mean(D)),
                                            psnr=float(np.mean(P)))
        print(f"  {src_tag} 过 {vtag:<16} lpips={np.mean(L):.4f} dists={np.mean(D):.4f} psnr={np.mean(P):.2f}", flush=True)

print("\n=== 保真度地板（对 GT 而言，越低越好）===")
print(f"{'路径':<26}{'LPIPS':>9}{'DISTS':>9}{'PSNR':>9}")
for k,v in rows.items(): print(f"{k:<26}{v['lpips']:>9.4f}{v['dists']:>9.4f}{v['psnr']:>9.2f}")
print("\n对照  run8 完整模型 lpips=0.3355 dists=0.2640 | VOSR零样本 0.3388/0.2248 | 原始LQ 0.4262/0.2942")
json.dump(rows, open(f"{R}/eval_out/vae_ceiling.json","w"), indent=2)
