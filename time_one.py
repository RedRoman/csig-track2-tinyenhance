"""单模型计时(独占卡上按序跑,绝对值可比):fold+channels_last+Inductor(MA-nocg)+手工 CUDA graph。用法: time_one.py 名 ckpt"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, torch.nn as nn, numpy as np
from pix_model import build_pix, fold_inference
DEV, DT = torch.device("cuda"), torch.float16
name, ck = sys.argv[1], sys.argv[2]
torch.manual_seed(0); x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)
class CL(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m.to(memory_format=torch.channels_last)
    def forward(s, t): return s.m(t.contiguous(memory_format=torch.channels_last))
m = build_pix(ck, device=DEV, dtype=torch.float32); p = sum(q.numel() for q in m.parameters()) / 1e6
fm = CL(fold_inference(m).to(DEV, DT).eval()).eval()
c = torch.compile(fm, mode="max-autotune-no-cudagraphs")
si = x.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for _ in range(3): so = c(si)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.no_grad(): so = c(si)
def run(q): si.copy_(q); g.replay(); return so.clone()
with torch.no_grad(): d = float((run(x).float() - fm(x).float()).abs().max())
def evt(n):
    ts = []
    with torch.no_grad():
        for _ in range(n):
            a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); run(x); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return ts
def wall(n):
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter(); run(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return ts
evt(50); E, Wl = [], []
for r in range(8): E += evt(100); Wl += wall(100)
E, Wl = np.array(E), np.array(Wl)
print(f"TIME {name:8s} {p:.2f}M max|d|={d:.4f} evt min {E.min():.3f} p10 {np.percentile(E,10):.3f} med {np.median(E):.3f} | wall min {Wl.min():.3f} p10 {np.percentile(Wl,10):.3f} med {np.median(Wl):.3f} ms", flush=True)
