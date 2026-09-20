"""剪解码器变体:从 G5c 部分初始化 → 折叠 → graph 计时;报告各变体 ms 与参数量(AutoDL 独占卡)。"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, torch.nn as nn, numpy as np
from pix_model import PixUNet2, fold_inference
DEV, DT = torch.device("cuda"), torch.float16
torch.manual_seed(0); x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)
sd = torch.load("/root/pix_out/pixG5c/last.pth", map_location="cpu", weights_only=False)["state"]
class CL(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m.to(memory_format=torch.channels_last)
    def forward(s, t): return s.m(t.contiguous(memory_format=torch.channels_last))
V = {"G5c ref (1,2,5)/(1,2)": dict(nbs=(1, 2, 5), nbs_dec=(1, 2)), "P1 (1,2,5)/(0,1)": dict(nbs=(1, 2, 5), nbs_dec=(0, 1)),
     "P2 (0,2,5)/(0,1)": dict(nbs=(0, 2, 5), nbs_dec=(0, 1)), "P3 (1,2,5)/(0,2)": dict(nbs=(1, 2, 5), nbs_dec=(0, 2)), "P4 (0,1,5)/(0,1)": dict(nbs=(0, 1, 5), nbs_dec=(0, 1))}
for name, kw in V.items():
    m = PixUNet2(chs=(32, 64, 128), norm="none", **kw); own = m.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}; m.load_state_dict(ok, strict=False)
    m = m.to(DEV).float(); p = sum(q.numel() for q in m.parameters()) / 1e6
    fm = CL(fold_inference(m).to(DEV, DT).eval()).eval()
    c = torch.compile(fm, mode="max-autotune-no-cudagraphs")
    si = x.clone(); st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st), torch.no_grad():
        for _ in range(3): so = c(si)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad(): so = c(si)
    def run(q): si.copy_(q); g.replay(); return so.clone()
    ts = []
    with torch.no_grad():
        for _ in range(50): run(x)
        torch.cuda.synchronize()
        for _ in range(300):
            a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); run(x); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    print(f"{name:24s} {p:.2f}M matched {len(ok)}/{len(own)}  graph p10 {np.percentile(ts,10):.3f} med {np.median(ts):.3f} ms", flush=True)
print("PRUNE_PROBE_DONE", flush=True)
