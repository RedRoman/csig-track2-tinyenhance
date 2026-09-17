"""沙箱友好编译档位实测:default 模式 + 单线程编译 + 冷缓存。"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/cd_ind_{int(time.time())}"
os.environ["TRITON_CACHE_DIR"] = f"/tmp/cd_tri_{int(time.time())}"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ.pop("TORCHDYNAMO_DISABLE", None)
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
torch.backends.cudnn.benchmark = False
DEV, DT = torch.device("cuda"), torch.float16
from tinysr_runtime import build_model
core, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth", device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
core.eval()
class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, t):
        o = s.m(t*0.5+0.5)
        if isinstance(o, tuple): o = o[0]
        return o*2-1
eager = W(core).to(DEV).eval()
x = (torch.rand(1,3,512,512,device=DEV,dtype=DT)*2-1)
def bench(fn, n=100, w=20):
    with torch.no_grad():
        for _ in range(w): fn(x)
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))
for mode in ("default", "reduce-overhead"):
    torch._dynamo.reset()
    cm = torch.compile(eager, mode=mode)
    t0 = time.time()
    with torch.no_grad(): cm(x).clone()
    print(f"[{mode} threads=1] 冷首编 {time.time()-t0:.0f}s  稳态 {bench(lambda t: cm(t).clone()):.3f} ms", flush=True)
print("COLD_DEFAULT_DONE", flush=True)
