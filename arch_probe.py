"""A100 上各架构折叠+channels_last+compile 的 512 单发时延,以及 fold 与原模型的数值等价性(用 L2d 真权重)。"""
import os, sys, time, copy, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, torch.nn as nn, numpy as np
from pix_model import PixUNet, PixUNet2, build_pix, fold_inference
DEV, DT = torch.device("cuda"), torch.float16
torch.manual_seed(0); x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)
def bench(fn, n=300, wu=50):
    with torch.no_grad():
        for _ in range(wu): fn(x)
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record(); fn(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))
class CL(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, t): return s.m(t.contiguous(memory_format=torch.channels_last))
def compiled(fm, mode):
    torch._dynamo.reset(); c = torch.compile(CL(fm).to(memory_format=torch.channels_last), mode=mode)
    with torch.no_grad():
        for _ in range(3): c(x)
    torch.cuda.synchronize(); return c
# 1) 真权重等价性
m = build_pix("/cache/zihaohe/csig_isr/train_out/pixL2d/last.pth", device=DEV, dtype=DT)
fm = fold_inference(m).to(DEV, DT).eval()
with torch.no_grad(): yb = m.core(x).clamp(-1, 1).float(); yf = fm(x).float()
print(f"L2d fold equiv: max|d|={float((yb-yf).abs().max()):.4f} mean|d|={float((yb-yf).abs().mean()):.6f}", flush=True)
c = compiled(fm, "reduce-overhead")
with torch.no_grad(): yc = c(x).float()
print(f"L2d fold+CL+RO vs eager: max|d|={float((yb-yc).abs().max()):.4f}  eager-orig {bench(lambda q: m.core(q).clamp(-1,1)):.3f} ms | fold eager {bench(fm):.3f} | fold+CL+RO {bench(lambda q: c(q).clone()):.3f} ms", flush=True)
c2 = compiled(fm, "max-autotune"); print(f"L2d fold+CL+MA {bench(lambda q: c2(q).clone()):.3f} ms", flush=True)
# 2) 新架构等价性(随机权重,ada/c2 非零以检验折叠)
def randomize(mod):
    with torch.no_grad():
        for n_, p in mod.named_parameters(): p.add_(torch.randn_like(p) * 0.02)
V = {"ref GN 2,2,2 (L2c)": dict(nbs=(2, 2, 2), norm="gn"), "N1 none 2,2,2": dict(nbs=(2, 2, 2), norm="none"),
     "N2 GN 1,2,2": dict(nbs=(1, 2, 2), norm="gn"), "N3 none 1,2,3": dict(nbs=(1, 2, 3), norm="none"),
     "N4 none 1,2,2": dict(nbs=(1, 2, 2), norm="none"), "N5 none 1,1,2": dict(nbs=(1, 1, 2), norm="none"),
     "N6 none 1,2,2 w48": dict(chs=(48, 96, 192), nbs=(1, 2, 2), norm="none")}
for name, kw in V.items():
    mm = PixUNet2(**{"chs": (32, 64, 128), **kw}).to(DEV).float(); randomize(mm); mm = mm.to(DT).eval()
    p = sum(q.numel() for q in mm.parameters()) / 1e6
    fm = fold_inference(mm).to(DEV, DT).eval()
    with torch.no_grad(): d = float((mm.core(x).clamp(-1, 1).float() - fm(x).float()).abs().max())
    c = compiled(fm, "reduce-overhead"); t = bench(lambda q: c(q).clone())
    print(f"{name:22s} {p:4.2f}M  fold max|d|={d:.4f}  fold+CL+RO {t:.3f} ms", flush=True)
    del mm, fm, c; torch.cuda.empty_cache()
print("ARCH_PROBE_DONE", flush=True)
