import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None)
sys.path.insert(0, "/cache/zihaohe")
import torch, numpy as np
from pix_model import build_pix
DEV, DT = torch.device("cuda"), torch.float16
torch.backends.cudnn.benchmark = False
m = build_pix("/cache/zihaohe/csig_isr/train_out/pixB2/best.pth", device=DEV, dtype=DT)
class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m.core(x).clamp(-1, 1)
w = W(m).eval()
x = (torch.rand(1,3,512,512,device=DEV,dtype=DT)*2-1)
def bench(fn, n=200, wu=30):
    with torch.no_grad():
        for _ in range(wu): fn(x)
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))
print(f"eager        {bench(lambda t: w(t)):.3f} ms", flush=True)
with torch.no_grad(): j = torch.jit.freeze(torch.jit.trace(w, x, check_trace=False).eval())
print(f"jit-freeze   {bench(lambda t: j(t)):.3f} ms", flush=True)
for mode in ("reduce-overhead", "max-autotune"):
    torch._dynamo.reset(); c = torch.compile(w, mode=mode); t0=time.time()
    with torch.no_grad(): c(x)
    print(f"compile {mode:16s} {bench(lambda t: c(t).clone()):.3f} ms  冷编译 {time.time()-t0:.0f}s", flush=True)
# 同卡 v8 参照(已知 14.97) ; 4K 整图内存峰值
torch.cuda.reset_peak_memory_stats(); big=(torch.rand(1,3,2160,3840,device=DEV,dtype=DT)*2-1)
with torch.no_grad(): y = w(big); torch.cuda.synchronize()
print(f"4K 整图 eager 峰值显存 {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
print("BENCH_DONE", flush=True)
