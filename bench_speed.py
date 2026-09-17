"""时延轴攻坚:全档位测速矩阵。官方协议:512x512 fp16,CUDA Event,20 warmup + 200 中位。

基线 = 提交件 jit 模型。候选 = eager(merge_lora) 及其 compile/graph 组合。
一致性以 jit 输出为参照(fp16 融合重排的小差异可接受,报数值供判断)。
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = ""
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
torch.backends.cudnn.benchmark = False
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

DEV, DT = torch.device("cuda"), torch.float16
MD = "/cache/zihaohe/csig_isr/submit/model_dir"
CK = "/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth"

def bench(fn, x, n=200, w=20):
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

# 1) jit 基线
jit = torch.jit.load(os.path.join(MD, "tinysr_x1.pt"), map_location=DEV); jit.eval()
with torch.no_grad(): y_ref = jit(x.clone())
t_jit = bench(lambda t: jit(t), x)
print(f"jit 基线            {t_jit:7.3f} ms", flush=True)

# 2) eager(merge_lora, [-1,1] 包装与 jit 同界面)
from tinysr_runtime import build_model
core, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
core.eval()
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, t):
        out = s.m(t * 0.5 + 0.5)
        if isinstance(out, tuple):
            out = out[0]
        return out * 2 - 1
eager = Wrap(core).to(DEV).eval()
with torch.no_grad(): y_e = eager(x.clone())
t_e = bench(lambda t: eager(t), x)
print(f"eager               {t_e:7.3f} ms   (权重=v8,jit=v2b,不比数值)", flush=True)

results = {"jit": t_jit, "eager": t_e}
# 3) compile 三档(逐一测,注意 reduce-overhead/max-autotune 自带 cudagraphs)
for mode in ("default", "reduce-overhead", "max-autotune"):
    try:
        torch._dynamo.reset()
        cm = torch.compile(eager, mode=mode)
        t0 = time.time()
        with torch.no_grad():
            y_c = cm(x.clone()).clone()
        compile_s = time.time() - t0
        t_c = bench(lambda t: cm(t).clone(), x)
        d = float((y_c - y_e).abs().max())
        print(f"compile {mode:<16}{t_c:7.3f} ms   |d|={d:.2e}   首编 {compile_s:.0f}s", flush=True)
        results[mode] = t_c
    except Exception as ex:
        print(f"compile {mode:<16}FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True)

# 4) eager + 手动 CUDA Graph
try:
    sIn = x.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3): eager(sIn)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        sOut = eager(sIn)
    def graph_fn(t):
        sIn.copy_(t); g.replay(); return sOut
    t_g = bench(graph_fn, x)
    print(f"eager+CUDAGraph     {t_g:7.3f} ms   |d|={float((graph_fn(x.clone()).clone()-y_e).abs().max()):.2e}", flush=True)
    results["graph"] = t_g
except Exception as ex:
    print(f"eager+CUDAGraph     FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True)

best = min(results, key=results.get)
print(f"\n最优 {best} {results[best]:.3f} ms  提速 {t_jit/results[best]:.3f}x   "
      f"榜乘数变化 {(100.15/results[best])**0.2/(100.15/11.31)**0.2:.3f}x", flush=True)
print("BENCH_DONE", flush=True)
