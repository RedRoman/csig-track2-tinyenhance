"""排查同一 jit 模型两处测速差一倍的原因。

可能性：
  A. runner.infer 里的 .to(device,dtype) 每次都做 H2D 拷贝/类型转换
  B. 预热不足
  C. 输入张量每次新建 vs 复用
  D. GPU 频率/其它进程干扰
"""
import os, sys, time
import torch
ROOT="/cache/zihaohe/csig_isr"; MD=f"{ROOT}/submit/model_dir"
dev=torch.device("cuda"); dt=torch.float16

m = torch.jit.load(f"{MD}/tinysr_x1.pt", map_location=dev); m.eval()
x_gpu = torch.rand(1,3,512,512, device=dev, dtype=dt)*2-1
x_cpu = (torch.rand(1,3,512,512)*2-1)

def bench(fn, inp, n=100, warm=30):
    with torch.no_grad():
        for _ in range(warm): fn(inp)
        torch.cuda.synchronize()
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(n): fn(inp)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

print(f"{'场景':<44}{'ms':>9}")
print("-"*54)
print(f"{'A. 直接调 jit（GPU 张量，无转换）':<44}{bench(lambda a: m(a), x_gpu):>9.3f}")
print(f"{'B. 每次 .to(dev,dtype)（GPU->GPU，应near-零）':<44}"
      f"{bench(lambda a: m(a.to(device=dev,dtype=dt)), x_gpu):>9.3f}")
print(f"{'C. 每次从 CPU 拷贝（H2D）':<44}"
      f"{bench(lambda a: m(a.to(device=dev,dtype=dt)), x_cpu):>9.3f}")

# 模拟 runner.infer 的完整调用
sys.path.insert(0, MD)
import importlib.util
spec=importlib.util.spec_from_file_location("r_", f"{MD}/runner.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
r=mod.Runner(MD)
print(f"{'D. runner.infer(GPU张量)':<44}{bench(lambda a: r.infer(a), x_gpu):>9.3f}")
print(f"{'E. runner.infer(CPU张量)':<44}{bench(lambda a: r.infer(a), x_cpu):>9.3f}")

# 检查是否有其它进程占 GPU
import subprocess
o=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory",
                  "--format=csv,noheader"],capture_output=True,text=True).stdout
print(f"\nGPU 上的进程：\n{o.strip() or '  仅本进程'}")
