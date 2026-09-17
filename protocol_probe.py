"""官方新协议反推:v8 在四种渲染协议下的全尺度 S,对照 85.23 的隐含值。
a=提交版(高斯窗+colorfix) b=高斯窗无colorfix c=平均窗无colorfix d=无重叠拼块
隐含参照:公式不变时 85.23 -> S=3.96(@8.68ms)/4.04(@11.31ms)"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from settings import VAL
from metrics_utils import create_metrics
from tinysr_runtime import build_model
from utils.wavelet_color_fix import wavelet_color_fix

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
M = create_metrics(DEV)
PAIRS = [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]

@torch.no_grad()
def tiled(model, x01, ov=256, gauss=True):
    tile=512; _,_,H,W = x01.shape; st = tile-ov if ov>0 else tile
    ys = list(range(0,max(H-tile,0)+1,st)) or [0]
    xs = list(range(0,max(W-tile,0)+1,st)) or [0]
    if ys[-1]+tile<H: ys.append(max(H-tile,0))
    if xs[-1]+tile<W: xs.append(max(W-tile,0))
    if gauss:
        t = torch.linspace(-1,1,tile,device=DEV); g = torch.exp(-(t**2)/(2*0.45**2))
        wm = (g[:,None]*g[None,:])[None,None].float()
    else:
        wm = torch.ones(1,1,tile,tile,device=DEV)
    out = torch.zeros(1,3,H,W,device=DEV); acc = torch.zeros(1,1,H,W,device=DEV)
    for y in ys:
        for xx in xs:
            with torch.autocast("cuda", dtype=DT):
                p = model(x01[:,:,y:y+tile,xx:xx+tile].to(DEV,dtype=DT)).float()
            out[:,:,y:y+tile,xx:xx+tile] += p*wm; acc[:,:,y:y+tile,xx:xx+tile] += wm
    return (out/acc.clamp_min(1e-8)).clamp(0,1)

model, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth",
                       device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
model.eval()
protos = [("a_提交版(overlap+colorfix)", dict(ov=256,gauss=True), True),
          ("b_overlap无colorfix",      dict(ov=256,gauss=True), False),
          ("c_平均窗无colorfix",       dict(ov=256,gauss=False), False),
          ("d_无重叠无colorfix",       dict(ov=0,gauss=False), False)]
for tag, kw, cf in protos:
    ss = []
    for lqf, gtf in PAIRS:
        lq = load(os.path.join(VAL,lqf)); gt = load(os.path.join(VAL,gtf)).to(DEV)
        sr = tiled(model, lq, **kw)
        if cf:
            f = wavelet_color_fix(ToPILImage()(sr[0].cpu()), ToPILImage()(lq[0].cpu()))
            sr = ToTensor()(f)[None].to(DEV)
        v = {}
        with torch.autocast("cuda", dtype=torch.float16):
            for k in ("lpips","dists"):
                v[k] = float(M[k](sr,gt)); torch.cuda.empty_cache()
        for k in ("niqe","musiq","maniqa","clipiqa"):
            v[k] = float(M[k](sr)); torch.cuda.empty_cache()
        ss.append((1-v["lpips"])+(1-v["dists"])+v["clipiqa"]+v["maniqa"]+v["musiq"]/100+max(0,(10-v["niqe"])/10))
    print(f"{tag:30s} S={np.mean(ss):.4f}", flush=True)
print("PROTO_DONE", flush=True)
