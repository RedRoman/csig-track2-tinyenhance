"""冷缓存 compile 首编时长(评测机语义)。每档一个子进程 + 全新 inductor/triton 缓存。"""
import os, subprocess, sys, time

CHILD = r'''
import os, sys, time, warnings; warnings.filterwarnings("ignore")
mode = sys.argv[1]
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/cold_ind_{mode}_{int(time.time())}"
os.environ["TRITON_CACHE_DIR"] = f"/tmp/cold_tri_{mode}_{int(time.time())}"
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
torch.backends.cudnn.benchmark = False
DEV, DT = torch.device("cuda"), torch.float16
from tinysr_runtime import build_model
core, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth",
                      device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
core.eval()
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, t):
        out = s.m(t * 0.5 + 0.5)
        if isinstance(out, tuple): out = out[0]
        return out * 2 - 1
eager = Wrap(core).to(DEV).eval()
x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)
cm = torch.compile(eager, mode=mode)
t0 = time.time()
with torch.no_grad():
    y = cm(x).clone()
print(f"[{mode}] 冷首编 {time.time()-t0:.0f}s", flush=True)
with torch.no_grad():
    for _ in range(20): cm(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(100):
        s0 = torch.cuda.Event(True); e0 = torch.cuda.Event(True)
        s0.record(); cm(x); e0.record(); torch.cuda.synchronize()
        ts.append(s0.elapsed_time(e0))
print(f"[{mode}] 稳态 {float(np.median(ts)):.3f} ms", flush=True)
'''
for mode in ("reduce-overhead", "max-autotune"):
    print(f"=== {mode} ===", flush=True)
    r = subprocess.run([sys.executable, "-c", CHILD, mode], capture_output=True, text=True,
                       env={**os.environ})
    for ln in r.stdout.splitlines():
        if ln.startswith("["): print(ln, flush=True)
    if r.returncode != 0:
        print("FAILED:", r.stderr[-300:], flush=True)
print("COLD_DONE", flush=True)
