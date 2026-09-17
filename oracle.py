"""架构天花板（oracle）：假设 DiT 完美——即它能把 LQ 的 latent 变成 GT 的 latent，
那么最终输出就是 decode(encode(GT))。给这个输出打完整六项分。

这是「换掉 DiT / 加层 / 换训练目标」这类改动的绝对上界。
对 TAESD（TinySR）和 SD2-VAE（VOSR/OSEDiff）各算一次，差值就是换 VAE 的价值上界。
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random
from PIL import Image
from settings import VAL, TS
from metrics_utils import create_metrics
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
GTS = ["case1_gt.png","case2_gt.png","case3_gt.jpg"]
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

sys.path.insert(0, os.path.join(TS)); from models.vae.autoencoder_tiny import AutoencoderTiny
tae = AutoencoderTiny.from_pretrained(os.path.join(TS,"checkpoint","vae","separable"),
                                      torch_dtype=DT).to(DEV,dtype=DT).eval()
def tae_rt(x01):
    sf = tae.config.scaling_factor
    z = tae.encode(x01*2-1).latents*sf
    return (tae.decode(z/sf, return_dict=False)[0]*0.5+0.5).clamp(0,1)
from diffusers import AutoencoderKL
kl = AutoencoderKL.from_pretrained(f"{R}/sd21base", subfolder="vae",
                                   torch_dtype=DT).to(DEV,dtype=DT).eval()
def kl_rt(x01):
    z = kl.encode(x01*2-1).latent_dist.mode()
    return (kl.decode(z, return_dict=False)[0]*0.5+0.5).clamp(0,1)

def score6(get):
    seed_all(); acc = {k:[] for k in ("lpips","dists","niqe","musiq","maniqa","clipiqa")}
    ps = []
    for g in GTS:
        gt = load(os.path.join(VAL,g)).to(DEV)
        y = get(g, gt)
        for k in acc:
            acc[k].append(float(M[k](y,gt) if k in ("lpips","dists") else M[k](y)))
        ps.append(10*np.log10(1.0/max(float(((y-gt)**2).mean()),1e-12)))
        del y, gt; torch.cuda.empty_cache()
    m = {k: float(np.mean(v)) for k,v in acc.items()}; m["psnr"]=float(np.mean(ps))
    m["score"] = ((1-m["lpips"])+(1-m["dists"])+m["clipiqa"]+m["maniqa"]
                  +m["musiq"]/100.0+max(0.0,(10-m["niqe"])/10.0))
    return m

rows = {}
rows["GT 本身（绝对上界）"]      = score6(lambda g,gt: gt)
rows["oracle: GT→TAESD→"]      = score6(lambda g,gt: rt_tiled(tae_rt, load(os.path.join(VAL,g))))
rows["oracle: GT→SD2-VAE→"]    = score6(lambda g,gt: rt_tiled(kl_rt,  load(os.path.join(VAL,g))))

print("\n=== 架构天花板 ===")
h = f"{'':<24}{'LPIPS':>8}{'DISTS':>8}{'NIQE':>8}{'MUSIQ':>8}{'MANIQA':>8}{'CLIPIQA':>9}{'PSNR':>8}{'Score':>9}"
print(h)
for k,v in rows.items():
    print(f"{k:<24}{v['lpips']:>8.4f}{v['dists']:>8.4f}{v['niqe']:>8.3f}{v['musiq']:>8.2f}"
          f"{v['maniqa']:>8.4f}{v['clipiqa']:>9.4f}{v['psnr']:>8.2f}{v['score']:>9.4f}")
print(f"\n对照  run12 实测 4.0025 | run8 3.8780 | TinySR零样本 3.8813 | VOSR零样本 3.5434 | OSEDiff 3.6999")
json.dump(rows, open(f"{R}/eval_out/oracle.json","w"), indent=2)
