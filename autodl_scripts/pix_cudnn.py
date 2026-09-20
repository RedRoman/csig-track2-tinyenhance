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
def bench(fn, xx, n=200, wu=40):
    with torch.no_grad():
        for _ in range(wu): fn(xx)
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(xx); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))
torch.backends.cudnn.benchmark = True
print(f"eager+benchmark        {bench(lambda t: w(t), x):.3f} ms", flush=True)
wc = w.to(memory_format=torch.channels_last); xc = x.contiguous(memory_format=torch.channels_last)
print(f"eager+benchmark+CL     {bench(lambda t: wc(t), xc):.3f} ms", flush=True)
with torch.no_grad(): j = torch.jit.freeze(torch.jit.trace(w, x, check_trace=False).eval())
print(f"jit+benchmark          {bench(lambda t: j(t), x):.3f} ms", flush=True)
with torch.no_grad(): jc = torch.jit.freeze(torch.jit.trace(wc, xc, check_trace=False).eval())
print(f"jit+benchmark+CL       {bench(lambda t: jc(t), xc):.3f} ms", flush=True)
torch._dynamo.reset(); c = torch.compile(w, mode="reduce-overhead")
with torch.no_grad(): c(x)
print(f"compile RO(参照)        {bench(lambda t: c(t).clone(), x):.3f} ms", flush=True)
print("CUDNN_DONE", flush=True)
