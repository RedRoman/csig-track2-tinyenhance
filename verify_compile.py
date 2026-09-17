"""compile 上线前三验:1) compile(jit) 是否可行(零依赖最优路径);
2) 冷缓存首编时长;3) 数值差异对全尺度 Score 的影响(3 张验证图 tiled 渲染对比)。"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/ind_cold_%d" % time.time()  # 冷缓存
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
torch.backends.cudnn.benchmark = False

DEV, DT = torch.device("cuda"), torch.float16
MD = "/cache/zihaohe/csig_isr/submit/model_dir"
Image.MAX_IMAGE_PIXELS = None

def bench(fn, x, n=100, w=20):
    with torch.no_grad():
        for _ in range(w): fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts))

torch.manual_seed(3)
x = (torch.rand(1, 3, 512, 512, device=DEV, dtype=DT) * 2 - 1)

# 验 1+2:compile(jit) 冷缓存
jit = torch.jit.load(os.path.join(MD, "tinysr_x1.pt"), map_location=DEV); jit.eval()
t_jit = bench(lambda t: jit(t), x)
print(f"jit 基线 {t_jit:.3f} ms", flush=True)
try:
    cj = torch.compile(jit, mode="max-autotune")
    t0 = time.time()
    with torch.no_grad():
        y = cj(x.clone()).clone()
    print(f"compile(jit) 冷首编 {time.time()-t0:.0f}s", flush=True)
    t_cj = bench(lambda t: cj(t).clone(), x)
    with torch.no_grad():
        d = float((cj(x.clone()).clone() - jit(x.clone())).abs().max())
    print(f"compile(jit,max-autotune) {t_cj:.3f} ms  提速 {t_jit/t_cj:.3f}x  |d|={d:.2e}", flush=True)
except Exception as ex:
    print(f"compile(jit) FAILED {type(ex).__name__}: {str(ex)[:150]}", flush=True)

print("VERIFY1_DONE", flush=True)
