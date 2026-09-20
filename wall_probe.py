"""墙钟(含 CPU 派发+同步)对比:OptimizedModule(reduce-overhead) vs 手工 CUDA graph 包 Inductor(no-cudagraphs) vs max-autotune。
在有训练争抢的卡上跑,报告 min 与 median;交错多轮以抵消漂移。"""
import os, sys, time, copy, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, torch.nn as nn, numpy as np
from pix_model import build_pix, fold_inference
DEV, DT = torch.device("cuda"), torch.float16
torch.manual_seed(0); x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)
class CL(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m.to(memory_format=torch.channels_last)
    def forward(s, t): return s.m(t.contiguous(memory_format=torch.channels_last))
m = build_pix("/cache/zihaohe/csig_isr/train_out/pixL2d/last.pth", device=DEV, dtype=torch.float32)
fm = CL(fold_inference(m).to(DEV, DT).eval()).eval()
with torch.no_grad(): ref = fm(x).float()
def mk_compiled(mode):
    torch._dynamo.reset(); c = torch.compile(fm, mode=mode)
    with torch.no_grad():
        for _ in range(4): c(x)
    torch.cuda.synchronize(); return c
A = mk_compiled("reduce-overhead"); fA = lambda q: A(q).clone()
C = mk_compiled("max-autotune"); fC = lambda q: C(q).clone()
# B: 手工 CUDA graph 包 no-cudagraphs 编译体
Bc = mk_compiled("max-autotune-no-cudagraphs")
static_in = x.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for _ in range(3): static_out = Bc(static_in)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.no_grad(): static_out = Bc(static_in)
def fB(q):
    static_in.copy_(q); g.replay(); return static_out.clone()
# D: 手工 CUDA graph 包 eager 折叠模型(无 Inductor 融合)
static_in2 = x.clone(); s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s2), torch.no_grad():
    for _ in range(3): static_out2 = fm(static_in2)
torch.cuda.current_stream().wait_stream(s2)
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2), torch.no_grad(): static_out2 = fm(static_in2)
def fD(q):
    static_in2.copy_(q); g2.replay(); return static_out2.clone()
with torch.no_grad():
    for nm, f in (("A RO", fA), ("B graph(MA-nocg)", fB), ("C MA", fC), ("D graph(eager)", fD)):
        print(f"{nm:18s} max|d| vs eager {float((f(x).float()-ref).abs().max()):.4f}", flush=True)
def wall(f, n=100):
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter(); f(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return ts
def evt(f, n=100):
    ts = []
    with torch.no_grad():
        for _ in range(n):
            a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); f(x); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return ts
res = {k: {"wall": [], "evt": []} for k in ("A RO", "B graph(MA-nocg)", "C MA", "D graph(eager)")}
fs = {"A RO": fA, "B graph(MA-nocg)": fB, "C MA": fC, "D graph(eager)": fD}
for k in fs: wall(fs[k], 30)
for r in range(12):
    for k, f in fs.items():
        res[k]["wall"] += wall(f, 60); res[k]["evt"] += evt(f, 60)
for k in fs:
    w, e = np.array(res[k]["wall"]), np.array(res[k]["evt"])
    print(f"{k:18s} wall min {w.min():.3f} p10 {np.percentile(w,10):.3f} med {np.median(w):.3f} | event min {e.min():.3f} p10 {np.percentile(e,10):.3f} med {np.median(e):.3f} ms", flush=True)
print("WALL_PROBE_DONE", flush=True)
