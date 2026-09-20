"""免 triton 运行时的加速路径:AOTInductor .so(运行时只需 libtorch)+ CUDA graph(空闲卡重测)。pixJ 权重,数值对 eager。"""
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
with torch.no_grad(): y_e = w(x)
print(f"eager        {bench(lambda t: w(t)):.2f} ms", flush=True)
# CUDA graph on eager (idle GPU)
sIn = x.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for _ in range(3): w(sIn)
torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.no_grad(): sOut = w(sIn)
def gf(t):
    sIn.copy_(t); g.replay(); return sOut.clone()
print(f"cudagraph    {bench(gf):.2f} ms  |d|={float((gf(x)-y_e).abs().max()):.2e}", flush=True)
# AOTI
t0=time.time()
with torch.no_grad():
    so = torch._export.aot_compile(w, (x,), options={"aot_inductor.output_path": "/cache/zihaohe/csig_isr/train_out/pixJ/pixJ_sm80.so", "max_autotune": False})
print(f"AOTI 编译 {time.time()-t0:.0f}s -> {so}", flush=True)
a = torch._export.aot_load(so, device="cuda")
with torch.no_grad(): y_a = a(x)
print(f"AOTI         {bench(lambda t: a(t)):.2f} ms  |d|={float((y_a-y_e).abs().max()):.2e}  mean|d|={float((y_a-y_e).abs().mean()):.2e}", flush=True)
# 真图检查(防 TinySR 那次的涂抹)
from PIL import Image
lq = torch.from_numpy(np.asarray(Image.open("/cache/zihaohe/csig_isr/data/赛题二/验证集/case1_lq.jpg").convert("RGB"), np.float32)/255.).permute(2,0,1)[None][:, :, :512, :512].to(DEV, DT)*2-1
with torch.no_grad(): d = (a(lq)-w(lq)).abs()
print(f"AOTI 真图 |d| p50={d.median():.2e} max={d.max():.2e}", flush=True)
torch._dynamo.reset(); c = torch.compile(w, mode="reduce-overhead")
with torch.no_grad(): c(x)
print(f"compile RO   {bench(lambda t: c(t).clone()):.2f} ms", flush=True)
print("AOTI_DONE", flush=True)
