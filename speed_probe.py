"""L2c 权重上的零重训加速探针:AdaGN 常量折叠 / channels_last / compile 模式 / cudnn.benchmark;并 profile kernel 分布。
用法: speed_probe.py [ckpt]   输出到 stdout。"""
import os, sys, time, math, copy, warnings; warnings.filterwarnings("ignore")
os.environ.pop("TORCHDYNAMO_DISABLE", None); sys.path.insert(0, "/cache/zihaohe")
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pix_model import PixUNet, build_pix
DEV, DT = torch.device("cuda"), torch.float16
CK = sys.argv[1] if len(sys.argv) > 1 else "/cache/zihaohe/csig_isr/train_out/pixL2c/last.pth"
torch.manual_seed(0)
x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)

def bench(fn, n=300, wu=50):
    with torch.no_grad():
        for _ in range(wu): fn(x)
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record(); fn(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))

class W(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, t): return s.m.core(t).clamp(-1, 1)

class FBlk(nn.Module):
    """AdaGN 常量(t 固定)折进 GroupNorm affine: GN(x)*(1+sc)+sh == GN'(x)."""
    def __init__(s, b, te):
        super().__init__()
        c = b.c1.in_channels
        s.n = nn.GroupNorm(8, c)
        with torch.no_grad():
            sc, sh = b.ada(te)[0].chunk(2, 0)
            s.n.weight.copy_(b.n.weight * (1 + sc)); s.n.bias.copy_(b.n.bias * (1 + sc) + sh)
        s.c1, s.c2 = b.c1, b.c2
    def forward(s, x): return x + s.c2(F.gelu(s.c1(F.gelu(s.n(x)))))

class Folded(nn.Module):
    def __init__(s, m):
        super().__init__()
        mf = copy.deepcopy(m).float()
        with torch.no_grad():
            t = torch.full((1,), mf.t_fixed, device=DEV, dtype=torch.float32)
            half = mf.tdim // 2
            fr = torch.exp(-math.log(1e4) * torch.arange(half, device=DEV, dtype=torch.float32) / half)
            te = mf.temb(torch.cat([torch.sin(t[:, None] * fr), torch.cos(t[:, None] * fr)], 1))
        s.ps = mf.ps; s.inc = mf.inc; s.down = mf.down; s.up = mf.up; s.outc = mf.outc
        s.enc = nn.ModuleList([nn.Sequential(*[FBlk(b, te) for b in blks]) for blks in mf.enc])
        s.mid = nn.Sequential(*[FBlk(b, te) for b in mf.mid])
        s.dec = nn.ModuleList([nn.Sequential(*[FBlk(b, te) for b in blks]) for blks in mf.dec])
    def forward(s, x):
        x0 = x
        _, _, H, W_ = x.shape; p = s.ps * 4
        ph, pw = (-H) % p, (-W_) % p
        if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        h = F.pixel_unshuffle(x, s.ps) if s.ps > 1 else x
        h = s.inc(h); skips = []
        for i, blks in enumerate(s.enc):
            h = blks(h); skips.append(h)
            if i < len(s.down): h = s.down[i](h)
        h = s.mid(h)
        for j, (up, blks) in enumerate(zip(s.up, s.dec)):
            h = blks(up(h) + skips[-2 - j])
        r = s.outc(h)
        if s.ps > 1: r = F.pixel_shuffle(r, s.ps)
        if ph or pw: r = r[:, :, :H, :W_]
        return (x0 + r).clamp(-1, 1)

m = build_pix(CK, device=DEV, dtype=DT)
print(f"ckpt {CK} cfg={m.cfg} params={sum(p.numel() for p in m.parameters())/1e6:.2f}M", flush=True)
base = W(m).eval()
fold = Folded(m).to(DEV, DT).eval()
with torch.no_grad():
    yb = base(x).float(); yf = fold(x).float()
print(f"fold numeric: max|d|={float((yb-yf).abs().max()):.5f} mean|d|={float((yb-yf).abs().mean()):.6f}", flush=True)
# 真图检查
try:
    from PIL import Image
    a = np.asarray(Image.open("/cache/zihaohe/csig_isr/data/赛题二/测试集/1.jpg").convert("RGB"), np.float32) / 255.
    xr = (torch.from_numpy(a[:512, :512]).permute(2, 0, 1)[None].to(DEV, DT) * 2 - 1)
    with torch.no_grad(): d = (base(xr).float() - fold(xr).float()).abs()
    print(f"fold numeric(real crop): max|d|={float(d.max()):.5f} mean|d|={float(d.mean()):.6f}", flush=True)
except Exception as ex: print("real check skipped:", ex)

print(f"eager base {bench(base):.3f} ms | eager fold {bench(fold):.3f} ms", flush=True)

def comp(mod, mode, cl=False, cudnn_bench=False):
    torch.backends.cudnn.benchmark = cudnn_bench
    torch._dynamo.reset()
    mm = copy.deepcopy(mod)
    if cl: mm = mm.to(memory_format=torch.channels_last)
    c = torch.compile(mm, mode=mode)
    xx = x.contiguous(memory_format=torch.channels_last) if cl else x
    t0 = time.time()
    with torch.no_grad():
        for _ in range(3): c(xx)
    torch.cuda.synchronize(); tc = time.time() - t0
    def fn(q):
        return c(q.contiguous(memory_format=torch.channels_last) if cl else q).clone()
    t = bench(fn)
    with torch.no_grad(): d = float((fn(x).float() - yb).abs().max())
    return t, tc, d

rows = [("base  reduce-overhead", base, "reduce-overhead", False, False),
        ("fold  reduce-overhead", fold, "reduce-overhead", False, False),
        ("fold  RO + cudnn.benchmark", fold, "reduce-overhead", False, True),
        ("fold  RO + channels_last", fold, "reduce-overhead", True, False),
        ("fold  max-autotune", fold, "max-autotune", False, False),
        ("fold  max-autotune + CL", fold, "max-autotune", True, False)]
for name, mod, mode, cl, cb in rows:
    try:
        t, tc, d = comp(mod, mode, cl, cb)
        print(f"{name:32s} {t:.3f} ms  (compile {tc:.0f}s, max|d| vs base {d:.4f})", flush=True)
    except Exception as ex:
        print(f"{name:32s} FAILED: {str(ex)[:200]}", flush=True)

# ---- kernel 分布(default 模式无 CUDA graph,便于按 kernel 归因) ----
from torch.profiler import profile, ProfilerActivity
torch.backends.cudnn.benchmark = False
for name, mod in (("base", base), ("fold", fold)):
    torch._dynamo.reset(); c = torch.compile(copy.deepcopy(mod), mode="default")
    with torch.no_grad():
        for _ in range(10): c(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20): c(x)
            torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.device_time_total > 0]
    tot = sum(e.device_time_total for e in ev) / 20
    nk = sum(e.count for e in ev) / 20
    print(f"\n#### profile {name} (default mode): {nk:.0f} kernels/iter, GPU busy {tot/1000:.3f} ms/iter", flush=True)
    agg = {}
    for e in ev:
        k = e.key
        cat = "conv" if ("conv" in k.lower() or "cudnn" in k.lower() or "gemm" in k.lower() or "implicit" in k.lower() or "wgrad" in k.lower()) else ("triton" if "triton" in k else "other")
        agg.setdefault(cat, [0, 0]); agg[cat][0] += e.device_time_total / 20; agg[cat][1] += e.count / 20
    for k, (tt, cc) in sorted(agg.items(), key=lambda z: -z[1][0]): print(f"   {k:8s} {tt/1000:.3f} ms  {cc:.0f} kernels", flush=True)
    for e in sorted(ev, key=lambda z: -z.device_time_total)[:14]:
        print(f"   {e.device_time_total/20/1000:.3f} ms x{e.count/20:.0f}  {e.key[:110]}", flush=True)
print("PROBE_DONE", flush=True)
