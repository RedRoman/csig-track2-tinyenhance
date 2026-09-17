import os, sys, importlib.util, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from PIL import Image
MD = "/cache/zihaohe/csig_isr/submit_v8/model_dir"
spec = importlib.util.spec_from_file_location("runner_v8", os.path.join(MD, "runner.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
r = mod.Runner(MD)
print("core:", type(r.core_model).__name__, flush=True)
Image.MAX_IMAGE_PIXELS = None
a = np.asarray(Image.open("/cache/zihaohe/csig_isr/data/赛题二/验证集/case1_lq.jpg").convert("RGB"), np.float32)/255.
lq = torch.from_numpy(a).permute(2,0,1)[None][:, :, :512, :512].cuda()
with torch.no_grad():
    y = r.infer((lq.half()*2-1)).float().cpu()
torch.save(y, "/cache/zihaohe/pair_runner.pt"); print("A_DONE", flush=True)
