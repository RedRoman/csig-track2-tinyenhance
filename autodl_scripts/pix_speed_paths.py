"""像素网各推理路径的 GPU 时间(同卡),用于判别理论 + 找不依赖 torch.compile 的加速路径。"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, numpy as np
from pix_model import build_pix
DEV, DT = torch.device("cuda"), torch.float16
m = build_pix("/cache/zihaohe/csig_isr/train_out/pixJ/best.pth", device=DEV, dtype=DT)
class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m.core(x).clamp(-1, 1)
w = W(m).eval(); x = (torch.rand(1,3,512,512,device=DEV,dtype=DT)*2-1)
def bench(fn, xx=x, n=200, wu=30):
    with torch.no_grad():
        for _ in range(wu): fn(xx)
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(xx); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))
def wall(fn, xx=x, n=100):
    with torch.no_grad():
        for _ in range(20): fn(xx)
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(n): y=fn(xx); torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1000
print(f"eager        event {bench(lambda t: w(t)):.2f}  wall {wall(lambda t: w(t)):.2f} ms", flush=True)
with torch.no_grad(): j = torch.jit.freeze(torch.jit.trace(w, x, check_trace=False).eval())
print(f"jit-freeze   event {bench(lambda t: j(t)):.2f}  wall {wall(lambda t: j(t)):.2f} ms", flush=True)
# jit + fuser 选项
torch._C._jit_set_profiling_executor(True); torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_gpu(True); torch._C._jit_set_texpr_fuser_enabled(True)
with torch.no_grad(): j2 = torch.jit.freeze(torch.jit.trace(w, x, check_trace=False).eval())
print(f"jit+nvfuser  event {bench(lambda t: j2(t)):.2f} ms", flush=True)
torch._dynamo.reset(); c = torch.compile(w, mode="reduce-overhead")
with torch.no_grad(): c(x)
print(f"compile RO   event {bench(lambda t: c(t).clone()):.2f}  wall {wall(lambda t: c(t).clone()):.2f} ms", flush=True)
# CPU 输入 -> GPU 输出的全流程(评测机可能这样计时)
xc = x.cpu()
print(f"compile RO   含H2D+D2H wall {wall(lambda t: c(t.cuda()).clone().cpu(), xc):.2f} ms", flush=True)
print(f"jit          含H2D+D2H wall {wall(lambda t: j(t.cuda()).cpu(), xc):.2f} ms", flush=True)
print("PATHS_DONE", flush=True)
