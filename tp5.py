"""亚毫秒架构编译时间(reduce-overhead,同卡),对照 T2 0.94。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, numpy as np
from pix_model import PixUNet
DEV, DT = torch.device("cuda"), torch.float16
x = (torch.rand(1,3,512,512,device=DEV,dtype=DT)*2-1)
def bench(fn, n=200, wu=30):
    with torch.no_grad():
        for _ in range(wu): fn(x)
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))
class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, t): return s.m.core(t).clamp(-1,1)
C = {"T2 ps2_32x3(ref)": dict(chs=(32,64,128), nb=3, ps=2),
     "ps2_32x2": dict(chs=(32,64,128), nb=2, ps=2), "ps2_24x3": dict(chs=(24,48,96), nb=3, ps=2), "ps2_24x2": dict(chs=(24,48,96), nb=2, ps=2),
     "ps2_16x3": dict(chs=(16,32,64), nb=3, ps=2), "ps2_32x3_2lv": dict(chs=(32,64), nb=3, ps=2), "ps2_32x2_2lv": dict(chs=(32,64), nb=2, ps=2),
     "ps4_48x3": dict(chs=(48,96,192), nb=3, ps=4), "ps4_64x2": dict(chs=(64,128,256), nb=2, ps=4), "ps4_48x2": dict(chs=(48,96,192), nb=2, ps=4)}
for name, kw in C.items():
    m = PixUNet(**kw).to(DEV, DT).eval(); p = sum(q.numel() for q in m.parameters())/1e6
    torch._dynamo.reset(); c = torch.compile(W(m).eval(), mode="reduce-overhead")
    with torch.no_grad(): c(x)
    print(f"{name:20s} {p:5.2f}M  compiled {bench(lambda q: c(q).clone()):.3f} ms", flush=True)
    del m, c; torch.cuda.empty_cache()
print("TP5_DONE", flush=True)
