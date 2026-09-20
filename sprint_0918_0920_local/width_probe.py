"""瓶颈加宽变体时延(fold+CL+graph),随机权重即可。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, torch.nn as nn, numpy as np
from pix_model import PixUNet2, fold_inference
DEV, DT = torch.device("cuda"), torch.float16
torch.manual_seed(0); x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)
class CL(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m.to(memory_format=torch.channels_last)
    def forward(s, t): return s.m(t.contiguous(memory_format=torch.channels_last))
V = {"G5c (32,64,128)x(1,2,5)": dict(chs=(32, 64, 128), nbs=(1, 2, 5)), "W1 (32,64,192)x(1,2,5)": dict(chs=(32, 64, 192), nbs=(1, 2, 5)),
     "W2 (32,64,256)x(1,2,5)": dict(chs=(32, 64, 256), nbs=(1, 2, 5)), "W3 (32,96,192)x(1,2,5)": dict(chs=(32, 96, 192), nbs=(1, 2, 5)),
     "W4 (32,64,192)x(1,2,4)": dict(chs=(32, 64, 192), nbs=(1, 2, 4))}
for name, kw in V.items():
    m = PixUNet2(norm="none", **kw).to(DEV).float()
    with torch.no_grad():
        for p in m.parameters(): p.add_(torch.randn_like(p) * 0.02)
    pn = sum(q.numel() for q in m.parameters()) / 1e6
    fm = CL(fold_inference(m).to(DEV, DT).eval()).eval(); c = torch.compile(fm, mode="max-autotune-no-cudagraphs")
    si = x.clone(); st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st), torch.no_grad():
        for _ in range(3): so = c(si)
    torch.cuda.current_stream().wait_stream(st); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad(): so = c(si)
    def run(q): si.copy_(q); g.replay(); return so.clone()
    ts = []
    with torch.no_grad():
        for _ in range(50): run(x)
        torch.cuda.synchronize()
        for _ in range(300):
            a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); run(x); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    print(f"{name:26s} {pn:.2f}M  graph p10 {np.percentile(ts,10):.3f} med {np.median(ts):.3f} ms", flush=True)
print("WIDTH_PROBE_DONE", flush=True)
