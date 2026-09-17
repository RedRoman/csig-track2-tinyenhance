"""导出折叠后的完整模块(pickle),绕开 TinySD3 懒折叠的重建问题。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import torch
from tinysr_runtime import build_model

DEV, DT = torch.device("cuda"), torch.float16
OUT = "/cache/zihaohe/csig_isr/submit_v8/model_dir"
core, rep = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth",
                        device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
core.eval()
# warmup 已在 build_model 内完成(折叠已触发)
torch.save(core.tr.cpu(), os.path.join(OUT, "tr_folded.pt"))
torch.save(core.vae.cpu(), os.path.join(OUT, "vae_folded.pt"))
torch.save({"ts": core.ts.cpu(), "pooled": core.pooled.cpu(),
            "alpha_gate": core.alpha_gate.cpu() if core.alpha_gate is not None else None,
            "scaling_factor": float(core.vae.config.scaling_factor)},
           os.path.join(OUT, "aux_folded.pt"))
for f in ("tr_folded.pt", "vae_folded.pt", "aux_folded.pt"):
    print(f, round(os.path.getsize(os.path.join(OUT, f))/1e6), "MB", flush=True)
os.remove(os.path.join(OUT, "tinysr_v8_assets.pt"))
print("FOLD_EXPORT_DONE", flush=True)
