import sys, warnings, inspect
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
warnings.filterwarnings("ignore")
import torch, tinysr_runtime as T
src = inspect.getsource(T.build_model)
i = src.find("core_refine")
print("build_model 里 core_refine 首次出现上下文:")
print(src[max(0,i-200):i+320])
