"""p 探针 checkpoint：恒等双步。

在 distill_v2b 的 payload 里写入 refine_steps=2, step_scale=[1.0, 0.0]。
第二步贡献乘 0 => 输出与单步逐位相同，但 DiT 前向两次 => 时延 ~x1.55。
同一批图、同一份权重、唯一变量是时延 —— 榜分差直接读出速度指数 p。

导出前后各做一次逐位一致性验证，不通过就拒绝产出。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import torch
from tinysr_runtime import build_model, checkpoint_payload

DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
SRC = f"{R}/train_out/distill_v2b/best.pth"
DST = f"{R}/train_out/distill_v2b/probe_2step.pth"

# 1) 载入原模型
m1, rep = build_model(SRC, device=DEV, dtype=DT, merge_lora=False,
                      alpha_mode="fixed", fixed_alpha=1.0)
m1.eval()
print("原模型载入:", rep["checkpoint"], "step", rep["step"], flush=True)

# 2) 开双步 + 恒等 step_scale
m1.enable_refine(2)
with torch.no_grad():
    m1.step_scale.copy_(torch.tensor([1.0, 0.0], device=m1.step_scale.device))
print("refine_steps =", m1.refine_steps, "step_scale =", m1.step_scale.tolist(), flush=True)

# 3) 保存 probe checkpoint（payload 会带上 refine_steps/step_scale）
payload = checkpoint_payload(m1, rep["step"], train_vae_decoder=False,
                             dense_names=[], args_dict={"probe": "identity-2step"})
torch.save(payload, DST)
print("已存", DST, flush=True)

# 4) 闭环验证：重新 build，对比单步模型输出逐位一致
m2, rep2 = build_model(DST, device=DEV, dtype=DT, merge_lora=False,
                       alpha_mode="fixed", fixed_alpha=1.0)
m2.eval()
print("probe 重载:", "refine_steps =", getattr(m2, "refine_steps", 1),
      "step_scale =", m2.step_scale.tolist() if hasattr(m2, "step_scale") else None, flush=True)

m0, _ = build_model(SRC, device=DEV, dtype=DT, merge_lora=False,
                    alpha_mode="fixed", fixed_alpha=1.0)
m0.eval()

torch.manual_seed(7)
x = torch.rand(1, 3, 512, 512, device=DEV, dtype=DT)
with torch.no_grad():
    y0 = m0(x)
    y2 = m2(x)
diff = float((y0 - y2).abs().max())
print(f"单步 vs 恒等双步 输出最大差 = {diff:.3e}", flush=True)
assert diff == 0.0, "非逐位一致，拒绝产出！"

# 5) 时延对比
import numpy as np
def bench(m, n=100, w=30):
    with torch.no_grad():
        for _ in range(w): m(x)
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); m(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts))

t0 = bench(m0); t2 = bench(m2)
print(f"单步 {t0:.2f} ms | 恒等双步 {t2:.2f} ms | 比值 {t2/t0:.3f}", flush=True)
print("PROBE_OK", flush=True)
