"""小壳时延矩阵:keep N 层 DiT + 完整 TAESD,compile max-autotune 后实测。
权重乱没关系,只测速度。0 层 = VAE 往返地板价。"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn as nn
torch.backends.cudnn.benchmark = False
DEV, DT = torch.device("cuda"), torch.float16
from tinysr_runtime import build_model

def bench(fn, x, n=150, w=25):
    with torch.no_grad():
        for _ in range(w): fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts))

x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)
print("baseline_11.31ms_jit  compiled_full=7.38ms(已知)", flush=True)
for keep in (12, 8, 6, 4, 2, 0):
    torch._dynamo.reset()
    core, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth",
                          device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
    core.eval()
    if keep < 12:
        blocks = core.tr.transformer_blocks
        core.tr.transformer_blocks = nn.ModuleList([blocks[i] for i in range(0, 12, max(1, 12//max(keep,1)))][:keep])
        if keep == 0:
            core.tr.transformer_blocks = nn.ModuleList([])
    class W(nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, t):
            o = s.m(t*0.5+0.5)
            if isinstance(o, tuple): o = o[0]
            return o*2-1
    m = torch.compile(W(core).to(DEV).eval(), mode="max-autotune")
    with torch.no_grad(): m(x)
    t = bench(lambda q: m(q), x)
    spd_mult = (7.38/t)**0.2 if t > 0 else 0
    print(f"keep={keep:2d}层  {t:6.3f} ms   vs现役乘数 {(100.15/t)**0.2/(100.15/8.68)**0.2:.3f}x", flush=True)
    del core, m; torch.cuda.empty_cache()
print("PROFILE_DONE", flush=True)
