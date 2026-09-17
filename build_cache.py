import os, sys, time, importlib.util
MD = "/cache/zihaohe/csig_isr/submit_v8b_wip"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(MD, "compile_cache", "inductor")
os.environ["TRITON_CACHE_DIR"] = os.path.join(MD, "compile_cache", "triton")
import torch, warnings; warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("rb", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0 = time.time()
r = m.Runner(MD)
print(f"冷编译+init {time.time()-t0:.0f}s core={type(r.core_model).__name__}", flush=True)
# 二次加载验证缓存命中
del r; torch._dynamo.reset()
t0 = time.time()
r2 = m.Runner(MD)
print(f"缓存命中 init {time.time()-t0:.0f}s core={type(r2.core_model).__name__}", flush=True)
x = (torch.rand(1,3,512,512,device="cuda",dtype=torch.float16)*2-1)
import numpy as np
with torch.no_grad():
    for _ in range(20): r2.infer(x)
    torch.cuda.synchronize(); ts=[]
    for _ in range(100):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); r2.infer(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
print(f"512 时延 {float(np.median(ts)):.3f} ms", flush=True)
# 4K tiled 路径冒烟
big = (torch.rand(1,3,2160,3840,device="cuda",dtype=torch.float16)*2-1)
t0=time.time(); y = r2.infer(big); torch.cuda.synchronize()
print(f"4K tiled 冒烟 {time.time()-t0:.1f}s out={tuple(y.shape)} range=({float(y.min()):.2f},{float(y.max()):.2f})", flush=True)
print("CACHE_BUILD_OK", flush=True)
