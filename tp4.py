"""候选模型编译后 GPU 时间(reduce-overhead, 同卡同轮), 含已训 ckpt 与未训架构。"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, numpy as np
from pix_model import PixUNet, build_pix
DEV, DT = torch.device("cuda"), torch.float16
T = "/cache/zihaohe/csig_isr/train_out"
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
def run(name, m):
    p = sum(q.numel() for q in m.parameters())/1e6
    torch._dynamo.reset(); c = torch.compile(W(m).eval(), mode="reduce-overhead")
    with torch.no_grad(): c(x)
    t = bench(lambda q: c(q).clone())
    print(f"{name:22s} {p:5.1f}M  compiled {t:.3f} ms  vs B2 x{t/REF:.3f}" if REF else f"{name:22s} {p:5.1f}M  compiled {t:.3f} ms", flush=True)
    del m, c; torch.cuda.empty_cache(); return t
REF = None
REF = run("pixB2(ckpt)", build_pix(f"{T}/pixB2/best.pth", device=DEV, dtype=DT))
run("pixS 48x3 ps2(ckpt)", build_pix(f"{T}/pixS/best.pth", device=DEV, dtype=DT))
for name, kw in {"ps2_48x2": dict(chs=(48,96,192), nb=2, ps=2), "ps2_40x3": dict(chs=(40,80,160), nb=3, ps=2), "ps2_32x3": dict(chs=(32,64,128), nb=3, ps=2),
                 "ps4_96x3": dict(chs=(96,192,384), nb=3, ps=4), "ps4_96x2": dict(chs=(96,192,384), nb=2, ps=4), "ps4_128x2": dict(chs=(128,256,512), nb=2, ps=4),
                 "ps4_64x3": dict(chs=(64,128,256), nb=3, ps=4)}.items():
    run(name, PixUNet(**kw).to(DEV, DT).eval())
print("TP4_DONE", flush=True)
