import sys, warnings
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
warnings.filterwarnings("ignore")
import torch, tinysr_runtime as T
print("模块:", T.__file__)
sd = T.read_checkpoint("/tmp/sm3/best.pth")
print("read_checkpoint 返回的键:", list(sd.keys()) if sd else None)
print("  refine_steps =", sd.get("refine_steps"))
print("  step_scale   =", sd.get("step_scale"))
