import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, numpy as np
from pix_model import PixUNet
DEV, DT = torch.device("cuda"), torch.float16
x = (torch.rand(1,3,512,512,device=DEV,dtype=DT)*2-1)
def bench(fn, n=150, wu=30):
    with torch.no_grad():
        for _ in range(wu): fn(x)
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))
for name, kw in {"ps2_64x3(pixB2)": dict(chs=(64,128,256), nb=3, ps=2), "ps2_80x3": dict(chs=(80,160,320), nb=3, ps=2), "ps2_64x4": dict(chs=(64,128,256), nb=4, ps=2), "ps2_96x3": dict(chs=(96,192,384), nb=3, ps=2), "ps2_64x3_4lv": dict(chs=(64,128,256,512), nb=3, ps=2)}.items():
    m = PixUNet(**kw).to(DEV, DT).eval(); p = sum(q.numel() for q in m.parameters())/1e6
    class W(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, t): return s.m.core(t).clamp(-1,1)
    torch._dynamo.reset(); c = torch.compile(W(m).eval(), mode="reduce-overhead")
    with torch.no_grad(): c(x)
    print(f"{name:18s} {p:6.1f}M  compiled {bench(lambda t: c(t).clone()):.2f} ms", flush=True)
    del m, c; torch.cuda.empty_cache()
print("TP3_DONE", flush=True)
