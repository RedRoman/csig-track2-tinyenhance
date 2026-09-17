"""Canonical tile-strategy sweep using the same TinySR/metric loader as v3."""
from __future__ import annotations
import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from settings import ROOT, VAL
from metrics_utils import create_metrics, proxy_score
from tinysr_runtime import build_model
from utils.wavelet_color_fix import wavelet_color_fix

DEV = torch.device("cuda")
DT = torch.float16
Image.MAX_IMAGE_PIXELS = None


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def w_gauss(n, ov, sigma=0.45):
    t = torch.linspace(-1, 1, n, device=DEV)
    return torch.exp(-(t**2)/(2*sigma**2))


def w_hann(n, ov):
    v = torch.ones(n, device=DEV)
    if ov > 0:
        t = torch.linspace(0, np.pi, 2*ov, device=DEV)
        e = (1-torch.cos(t))/2
        v[:ov] = e[:ov]
        v[-ov:] = e[ov:].flip(0)
    return v


@torch.no_grad()
def tiled(model, img01, tile, ov, wfn, pad_mode):
    _, _, H0, W0 = img01.shape
    pad = ov//2 if pad_mode else 0
    if pad:
        img01 = F.pad(img01, (pad,pad,pad,pad), mode="reflect")
    _, _, H, W = img01.shape
    st = tile-ov
    ys = list(range(0, max(H-tile,0)+1, st)); xs = list(range(0, max(W-tile,0)+1, st))
    if not ys or ys[-1]+tile < H: ys.append(max(H-tile,0))
    if not xs or xs[-1]+tile < W: xs.append(max(W-tile,0))
    wy, wx = wfn(tile,ov), wfn(tile,ov)
    wm = (wy[:,None]*wx[None,:])[None,None].float()
    out = torch.zeros(1,3,H,W,device=DEV,dtype=torch.float32)
    acc = torch.zeros(1,1,H,W,device=DEV,dtype=torch.float32)
    for y in ys:
        for x in xs:
            with torch.autocast(device_type="cuda", dtype=DT):
                o = model(img01[:,:,y:y+tile,x:x+tile].to(DEV,dtype=DT)).float()
            out[:,:,y:y+tile,x:x+tile] += o*wm
            acc[:,:,y:y+tile,x:x+tile] += wm
    r = (out/acc.clamp_min(1e-8)).clamp(0,1)
    return r[:,:,pad:pad+H0,pad:pad+W0] if pad else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alpha_mode", choices=["auto","fixed","adaptive"], default="auto")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--overlaps", default="64,128,192,256")
    ap.add_argument("--out", default=os.path.join(ROOT,"eval_out","v3_tile_sweep.json"))
    args = ap.parse_args()

    model, report = build_model(args.ckpt, device=DEV, dtype=DT, merge_lora=True,
                                alpha_mode=args.alpha_mode, fixed_alpha=args.alpha)
    model.eval(); print("load report:", report)

    pairs=[]
    for lqn,gtn in [("case1_lq.jpg","case1_gt.png"),("case2_lq.jpg","case2_gt.png"),("case3_lq.jpg","case3_gt.jpg")]:
        lq,gt=load_img(os.path.join(VAL,lqn)),load_img(os.path.join(VAL,gtn))
        if lq.shape!=gt.shape: continue
        _,_,H,W=gt.shape; y,x=max(0,H//2-512),max(0,W//2-512)
        pairs.append((lq[:,:,y:y+1024,x:x+1024].clone(),gt[:,:,y:y+1024,x:x+1024].clone()))
    M=create_metrics(DEV,include_psnr=True)
    rows=[]
    for ov,wname,padm in itertools.product([int(x) for x in args.overlaps.split(",")], ["gauss","hann"],[False,True]):
        res={k:[] for k in M}; wfn=w_gauss if wname=="gauss" else w_hann
        for lq,gt in pairs:
            o=tiled(model,lq,512,ov,wfn,padm)
            fixed=wavelet_color_fix(ToPILImage()(o[0].cpu()),ToPILImage()(lq[0].cpu()))
            o=ToTensor()(fixed)[None].to(DEV); g=gt.to(DEV)
            for k,fn in M.items():
                v=fn(o,g) if k in ("lpips","dists","psnr") else fn(o)
                res[k].append(float(v.item()))
            torch.cuda.empty_cache()
        m={k:float(np.mean(v)) for k,v in res.items()}
        sc=proxy_score(m)
        row={"overlap":ov,"weight":wname,"pad":padm,"score":sc,"metrics":m}
        rows.append(row)
        print(f"ov={ov:3d} {wname:5s} pad={padm} score={sc:.4f} PSNR={m['psnr']:.3f}")
    best=max(rows,key=lambda x:x["score"])
    payload={"checkpoint":args.ckpt,"alpha_mode":args.alpha_mode,"alpha":args.alpha,"best":best,"rows":rows}
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,"w") as f: json.dump(payload,f,indent=2)
    print("best",best)

if __name__=="__main__":
    main()
