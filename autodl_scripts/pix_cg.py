import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe")
import torch, numpy as np
from pix_model import build_pix
DEV, DT = torch.device("cuda"), torch.float16
m = build_pix("/cache/zihaohe/csig_isr/train_out/pixB2/best.pth", device=DEV, dtype=DT)
class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m.core(x).clamp(-1, 1)
w = W(m).eval(); x = (torch.rand(1,3,512,512,device=DEV,dtype=DT)*2-1)
def bench(fn, n=200, wu=30):
    with torch.no_grad():
        for _ in range(wu): fn(x)
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))
print(f"eager {bench(lambda t: w(t)):.3f} ms", flush=True)
sIn = x.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for _ in range(3): w(sIn)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.no_grad(): sOut = w(sIn)
def gf(t):
    sIn.copy_(t); g.replay(); return sOut.clone()
with torch.no_grad(): d = float((gf(x) - w(x)).abs().max())
print(f"cudagraph(eager) {bench(gf):.3f} ms  |d|={d:.2e}", flush=True)
with torch.no_grad(): j = torch.jit.freeze(torch.jit.trace(w, x, check_trace=False).eval())
for _ in range(5): j(x)
sIn2 = x.clone(); s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s2), torch.no_grad():
    for _ in range(3): j(sIn2)
torch.cuda.current_stream().wait_stream(s2)
try:
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2), torch.no_grad(): sOut2 = j(sIn2)
    def gf2(t):
        sIn2.copy_(t); g2.replay(); return sOut2.clone()
    print(f"cudagraph(jit) {bench(gf2):.3f} ms", flush=True)
except Exception as ex:
    print("cudagraph(jit) FAILED", type(ex).__name__, flush=True)
print("CG_DONE", flush=True)
