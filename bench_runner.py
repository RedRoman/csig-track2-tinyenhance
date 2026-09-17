"""新旧 runner 同机对测：时延 + 输出一致性。

官方协议同款：512x512 fp16 单帧，CUDA Event 计时，20 warmup + 200 次取中位。
一致性：CUDA Graph 回放与普通前向逐位比较（应为 0，同 kernel 同顺序）。
"""
import os, sys, importlib.util, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch

MD = "/cache/zihaohe/csig_isr/submit/model_dir"
NEW_RUNNER = "/cache/zihaohe/runner_cudagraph.py"


def load_runner(py_path, model_dir):
    spec = importlib.util.spec_from_file_location("runner_mod_" + os.path.basename(py_path), py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Runner(model_dir)


def bench(fn, x, n=200, w=20):
    with torch.no_grad():
        for _ in range(w):
            fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts)), float(np.percentile(ts, 90))


torch.manual_seed(3)
x = (torch.rand(1, 3, 512, 512, device="cuda", dtype=torch.float16) * 2 - 1)
x2 = (torch.rand(1, 3, 512, 512, device="cuda", dtype=torch.float16) * 2 - 1)  # 预生成，避免 capture 失败污染 RNG 状态

print("=== 旧 runner（当前提交所用）===", flush=True)
r_old = load_runner(os.path.join(MD, "runner.py"), MD)
y_old = r_old.infer(x.clone())
m, p90 = bench(lambda t: r_old.infer(t), x)
print(f"旧: 中位 {m:.3f} ms  p90 {p90:.3f} ms", flush=True)
del r_old; torch.cuda.empty_cache()

print("=== 新 runner（CUDA Graph）===", flush=True)
r_new = load_runner(NEW_RUNNER, MD)
y_new = r_new.infer(x.clone())
m2, p902 = bench(lambda t: r_new.infer(t), x)
print(f"新: 中位 {m2:.3f} ms  p90 {p902:.3f} ms  (graph={'on' if r_new._graph is not None else 'OFF!'})", flush=True)

diff = float((y_old - y_new).abs().max())
print(f"输出一致性 |old-new| max = {diff:.3e}", flush=True)
print(f"提速: {m/m2:.3f}x   官方加速比 {100.15/m:.2f} -> {100.15/m2:.2f}", flush=True)
# 连续两次调用（不同输入）验证 graph 回放正确性
y2a = r_new.infer(x2.clone())
r_dummy = load_runner(os.path.join(MD, "runner.py"), MD)
y2b = r_dummy.infer(x2.clone())
print(f"第二输入一致性 = {float((y2a - y2b).abs().max()):.3e}", flush=True)
print("BENCH_OK", flush=True)
