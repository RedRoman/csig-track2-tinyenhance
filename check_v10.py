import os, sys, time, importlib.util, warnings; warnings.filterwarnings("ignore")
import torch, numpy as np
MD = sys.argv[1]
spec = importlib.util.spec_from_file_location("rv", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0 = time.time(); r = m.Runner(MD)
print(f"init {time.time()-t0:.0f}s core={type(r.core_model).__name__}", flush=True)
x = (torch.rand(1,3,512,512,device="cuda",dtype=torch.float16)*2-1)
with torch.no_grad():
    for _ in range(20): r.infer(x)
    torch.cuda.synchronize(); ts=[]
    for _ in range(100):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); r.infer(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
print(f"512 median {float(np.median(ts)):.3f} ms", flush=True)
big = (torch.rand(1,3,2160,3840,device="cuda",dtype=torch.float16)*2-1)
t0=time.time(); y = r.infer(big); torch.cuda.synchronize()
print(f"4K tiled {time.time()-t0:.1f}s {tuple(y.shape)}", flush=True)
print("CHECK_V10_OK", flush=True)
