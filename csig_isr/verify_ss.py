"""验证 step_scale 存取闭合：训完加载回来步长必须保住，否则多步细化白训。"""
import sys, warnings
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
warnings.filterwarnings("ignore")
import torch
from tinysr_runtime import build_model
m, rep = build_model("/tmp/sm3/best.pth", device=torch.device("cuda"),
                     dtype=torch.float16, merge_lora=False,
                     alpha_mode="fixed", fixed_alpha=1.0)
print("load report refine:", rep.get("refine_steps"), rep.get("step_scale"))
print("模型层面 refine_steps =", getattr(m, "refine_steps", 1))
x = torch.rand(1, 3, 512, 512, device="cuda", dtype=torch.float16)
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = m(x)
print("前向 OK", tuple(y.shape))
