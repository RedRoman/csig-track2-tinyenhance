"""像素空间一步式去噪小网时延探针(512², fp16, compile max-autotune, A100)。
带时间步嵌入(AdaGN 调制)以满足 diffusion 形态;输出残差。"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None)
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
DEV, DT = torch.device("cuda"), torch.float16
torch.backends.cudnn.benchmark = False

class Blk(nn.Module):
    def __init__(s, c, tdim):
        super().__init__(); s.c1 = nn.Conv2d(c, c, 3, padding=1); s.c2 = nn.Conv2d(c, c, 3, padding=1)
        s.n = nn.GroupNorm(8, c); s.ada = nn.Linear(tdim, 2*c)
    def forward(s, x, te):
        sc, sh = s.ada(te)[:, :, None, None].chunk(2, 1)
        h = s.n(x) * (1 + sc) + sh
        return x + s.c2(F.gelu(s.c1(F.gelu(h))))
class TinyUNet(nn.Module):
    def __init__(s, chs=(32, 64, 128), nb=2, tdim=64):
        super().__init__(); s.tdim = tdim
        s.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        s.inc = nn.Conv2d(3, chs[0], 3, padding=1)
        s.enc = nn.ModuleList(); s.down = nn.ModuleList()
        for i, c in enumerate(chs):
            s.enc.append(nn.ModuleList([Blk(c, tdim) for _ in range(nb)]))
            if i < len(chs)-1: s.down.append(nn.Conv2d(c, chs[i+1], 3, stride=2, padding=1))
        s.mid = nn.ModuleList([Blk(chs[-1], tdim) for _ in range(nb)])
        s.up = nn.ModuleList(); s.dec = nn.ModuleList()
        for i in range(len(chs)-1, 0, -1):
            s.up.append(nn.ConvTranspose2d(chs[i], chs[i-1], 4, stride=2, padding=1))
            s.dec.append(nn.ModuleList([Blk(chs[i-1], tdim) for _ in range(nb)]))
        s.outc = nn.Conv2d(chs[0], 3, 3, padding=1)
    def forward(s, x, t):
        half = s.tdim // 2
        fr = torch.exp(-np.log(1e4) * torch.arange(half, device=x.device, dtype=x.dtype) / half)
        te = s.temb(torch.cat([torch.sin(t[:, None]*fr), torch.cos(t[:, None]*fr)], 1))
        h = s.inc(x); skips = []
        for i, blks in enumerate(s.enc):
            for b in blks: h = b(h, te)
            skips.append(h)
            if i < len(s.down): h = s.down[i](h)
        for b in s.mid: h = b(h, te)
        for j, (up, blks) in enumerate(zip(s.up, s.dec)):
            h = up(h) + skips[-2-j]
            for b in blks: h = b(h, te)
        return x + s.outc(h)

def bench(fn, n=150, w=30):
    x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT)*2-1); t = torch.full((1,), 999., device=DEV, dtype=DT)
    with torch.no_grad():
        for _ in range(w): fn(x, t)
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record(); fn(x, t); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))

CANDS = {"u32x2_3lv": dict(chs=(32,64,128), nb=2), "u48x2_3lv": dict(chs=(48,96,192), nb=2),
         "u64x2_3lv": dict(chs=(64,128,256), nb=2), "u32x3_4lv": dict(chs=(32,64,128,256), nb=3),
         "u48x3_4lv": dict(chs=(48,96,192,384), nb=3), "u64x3_4lv": dict(chs=(64,128,256,512), nb=3)}
print(f"{'model':12s} {'params(M)':>9s} {'eager':>8s} {'compiled':>9s}", flush=True)
for name, kw in CANDS.items():
    m = TinyUNet(**kw).to(DEV, DT).eval()
    p = sum(q.numel() for q in m.parameters())/1e6
    te = bench(lambda x, t: m(x, t))
    torch._dynamo.reset(); cm = torch.compile(m, mode="max-autotune")
    tc = bench(lambda x, t: cm(x, t).clone())
    print(f"{name:12s} {p:9.2f} {te:8.2f} {tc:9.2f}", flush=True)
print("PROBE_DONE", flush=True)
