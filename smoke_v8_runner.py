"""新 runner 冒烟:官方协议 bench + 真图数值 vs eager + 路径确认。"""
import os, sys, time, importlib.util, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from PIL import Image

MD = "/cache/zihaohe/csig_isr/submit_v8/model_dir"
spec = importlib.util.spec_from_file_location("runner_v8", os.path.join(MD, "runner.py"))
mod = importlib.util.module_from_spec(spec)
t0 = time.time()
spec.loader.exec_module(mod)
r = mod.Runner(MD)
print(f"init 总耗时 {time.time()-t0:.0f}s  core={type(r.core_model).__name__}", flush=True)

torch.manual_seed(3)
x = (torch.rand(1, 3, 512, 512, device="cuda", dtype=torch.float16) * 2 - 1)
def bench(fn, x, n=200, w=20):
    with torch.no_grad():
        for _ in range(w): fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts)), float(np.percentile(ts, 90))
m, p90 = bench(lambda t: r.infer(t), x)
print(f"新 runner: 中位 {m:.3f} ms  p90 {p90:.3f} ms", flush=True)

# 真图数值对拍 eager
for m in list(sys.modules):
    if m == "tinysr_runtime":
        del sys.modules[m]
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("tinysr_runtime_full", "/cache/zihaohe/csig_isr/v3/code/tinysr_runtime.py")
_full = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_full)
build_model = _full.build_model
from settings import VAL
core, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth",
                      device=torch.device("cuda"), dtype=torch.float16,
                      merge_lora=True, alpha_mode="auto")
core.eval()
Image.MAX_IMAGE_PIXELS = None
a = np.asarray(Image.open(os.path.join(VAL, "case1_lq.jpg")).convert("RGB"), np.float32)/255.
lq = torch.from_numpy(a).permute(2,0,1)[None][:, :, :512, :512].cuda()
with torch.no_grad():
    y_r = r.infer((lq.half()*2-1)).float()
    y_e = core(lq.half())
    if isinstance(y_e, tuple): y_e = y_e[0]
    y_e = (y_e.float()*2-1)
d = (y_r - y_e).abs()
print(f"真图 vs eager: p50={d.median():.5f} max={d.max():.4f}", flush=True)
print("SMOKE_OK", flush=True)
