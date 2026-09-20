"""同卡交错计时:多个 ckpt 各建 fold+channels_last+Inductor(MA-nocg)+手工 CUDA graph,轮转 benchmark,报告相对第一个的时延比。
用法: time_pair.py 名=ckpt 名=ckpt ...   (第一个为参照,通常 L2d)"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, torch.nn as nn, numpy as np
torch._dynamo.config.cache_size_limit = 64   # 多模型同一 forward 代码对象会重编译多次
from pix_model import build_pix, fold_inference
DEV, DT = torch.device("cuda"), torch.float16
torch.manual_seed(0); x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)
class CL(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m.to(memory_format=torch.channels_last)
    def forward(s, t): return s.m(t.contiguous(memory_format=torch.channels_last))
def make(ck):
    m = build_pix(ck, device=DEV, dtype=torch.float32)
    fm = CL(fold_inference(m).to(DEV, DT).eval()).eval()
    c = torch.compile(fm, mode="max-autotune-no-cudagraphs")   # 不能 reset:reset 会释放已捕获 CUDA graph 依赖的编译体缓冲 → illegal memory access
    si = x.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3): so = c(si)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad(): so = c(si)
    def run(q): si.copy_(q); g.replay(); return so.clone()
    with torch.no_grad(): d = float((run(x).float() - fm(x).float()).abs().max())
    return run, d, sum(p.numel() for p in m.parameters()) / 1e6
specs = [a.split("=", 1) for a in sys.argv[1:]]
fns = {}
for name, ck in specs:
    fns[name], d, p = make(ck); print(f"built {name:10s} {p:.2f}M  graph vs eager max|d|={d:.4f}", flush=True)
def evt(f, n=80):
    ts = []
    with torch.no_grad():
        for _ in range(n):
            a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); f(x); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return ts
def wall(f, n=80):
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter(); f(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return ts
res = {k: {"e": [], "w": []} for k in fns}
for k in fns: evt(fns[k], 30)
for r in range(10):
    for k, f in fns.items(): res[k]["e"] += evt(f); res[k]["w"] += wall(f)
ref = specs[0][0]
def stat(a): a = np.array(a); return a.min(), np.percentile(a, 10), np.median(a)
rm = stat(res[ref]["e"]); rw = stat(res[ref]["w"])
print(f"\n{'name':10s} {'evt min':>8s} {'evt p10':>8s} {'evt med':>8s} | {'wall min':>8s} {'wall p10':>8s} | ratio(evt p10) ratio(wall min)")
for k in fns:
    e = stat(res[k]["e"]); w = stat(res[k]["w"])
    print(f"{k:10s} {e[0]:8.3f} {e[1]:8.3f} {e[2]:8.3f} | {w[0]:8.3f} {w[1]:8.3f} | {e[1]/rm[1]:8.3f}      {w[0]/rw[0]:8.3f}", flush=True)
print("TIME_PAIR_DONE", flush=True)
