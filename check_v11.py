"""v11 校验:编译路径 + 污染检测(sys.path/sys.modules 必须干净,评测环境的包不能被劫持)。"""
import os, sys, time, importlib.util, warnings; warnings.filterwarnings("ignore")
import torch, numpy as np
MD = sys.argv[1]
sp_before = list(sys.path); mods_before = set(sys.modules)
spec = importlib.util.spec_from_file_location("rv", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0 = time.time(); r = m.Runner(MD)
print(f"init {time.time()-t0:.0f}s core={type(r.core_model).__name__}", flush=True)
# 污染检测
n0 = len(sp_before)
prepended = [p for p in sys.path[:n0] if p not in sp_before[:n0]]
ours = [p for p in sys.path if "/_work" in p or "tinysr_" in p]
leak_mods = [k for k in sys.modules if k == "models" or k.startswith("models.")]
import requests, typing_extensions
ok_req = all("/_work" not in m.__file__ and "tinysr_" not in m.__file__ for m in (requests, typing_extensions))
print(f"污染检测: 前插={prepended} 我方追加={ours} models包残留={leak_mods} 评测环境包未被劫持={ok_req}", flush=True)
assert not prepended and not leak_mods and ok_req, "POLLUTION"
x = (torch.rand(1,3,512,512,device="cuda",dtype=torch.float16)*2-1)
with torch.no_grad():
    for _ in range(20): r.infer(x)
    torch.cuda.synchronize(); ts=[]
    for _ in range(100):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); r.infer(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
print(f"512 median {float(np.median(ts)):.3f} ms", flush=True)
big = (torch.rand(1,3,2160,3840,device="cuda",dtype=torch.float16)*2-1)
t0=time.time(); y = r.infer(big); torch.cuda.synchronize()
y2 = r.infer(big[:, :, :1080, :1920]); torch.cuda.synchronize()
print(f"4K tiled {time.time()-t0:.1f}s {tuple(y.shape)}  1080p {tuple(y2.shape)}", flush=True)
print("CHECK_V11_OK", flush=True)
