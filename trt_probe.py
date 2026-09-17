"""TensorRT 可行性探针：把现有 distill_v2b 直接压给 torch_tensorrt。

要回答三件事:
  1) 能不能编译成功（TinySR 的 SD3 DiT 有 SDPA/RoPE/GroupNorm 等，TRT 未必吃）
  2) 编译后时延（fp16）
  3) 数值一致性（相对 eager 的 max/mean diff，超过 1e-3 蒸馏保真度会掉）

编译 fp16，官方评测环境也是 fp16，最合规。
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from tinysr_runtime import build_model
import torch_tensorrt

DEV, DT = torch.device("cuda"), torch.float16
CK = "/cache/zihaohe/csig_isr/train_out/distill_v2b/best.pth"

# 1) 加载现役学生，合并 LoRA
model, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=True,
                       alpha_mode="fixed", fixed_alpha=1.0)
model.eval()

x = torch.rand(1, 3, 512, 512, device=DEV, dtype=DT)


def bench(fn, n=100, w=30):
    with torch.no_grad():
        for _ in range(w): fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts))


# 2) eager 基准
with torch.no_grad():
    y_eager = model(x).float()
t_eager = bench(lambda x: model(x))
print(f"eager fp16: {t_eager:.2f} ms  (输出范围 [{float(y_eager.min()):.3f},{float(y_eager.max()):.3f}])",
      flush=True)

# 3) 常规 jit trace（对照，这就是我们现役提交的路径）
with torch.no_grad():
    jit_model = torch.jit.trace(model, x, strict=False)
    jit_model = torch.jit.optimize_for_inference(jit_model)
t_jit = bench(lambda x: jit_model(x))
with torch.no_grad():
    y_jit = jit_model(x).float()
print(f"jit  fp16: {t_jit:.2f} ms  |y-y_eager|max={float((y_jit-y_eager).abs().max()):.2e}",
      flush=True)

# 4) torch_tensorrt 编译
print("=== 编译 TRT ===", flush=True)
t0 = time.time()
try:
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(shape=(1, 3, 512, 512), dtype=torch.float16)],
        enabled_precisions={torch.float16},
        ir="dynamo",
        min_block_size=1,
        truncate_double=True,
        require_full_compilation=False,
    )
    print(f"  编译耗时 {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  编译失败: {type(e).__name__}: {str(e)[:400]}", flush=True)
    sys.exit(0)

# 5) 数值 + 时延
with torch.no_grad():
    for _ in range(30): trt_model(x)  # warmup（TRT engine 有 lazy init）
    y_trt = trt_model(x).float()
diff = (y_trt - y_eager).abs()
print(f"trt fp16 数值: max={float(diff.max()):.2e} mean={float(diff.mean()):.2e}", flush=True)
t_trt = bench(lambda x: trt_model(x))
print(f"trt fp16 时延: {t_trt:.2f} ms  加速比(vs eager) {t_eager/t_trt:.2f}x", flush=True)

# 6) 榜分预算（用已验证公式）
def score_bump(t_new, t_old, S=4.0577):
    return 3.357 * 2**S * (100.15/t_new)**0.2, 3.357 * 2**S * (100.15/t_old)**0.2

for tag, t in [("eager", t_eager), ("jit", t_jit), ("trt", t_trt)]:
    s = 3.357 * 2**4.0577 * (100.15/t)**0.2
    print(f"  {tag}: {t:.2f}ms → 预测榜分 {s:.1f}", flush=True)
