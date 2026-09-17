"""导出 v8 自包含推理资产:merged fp16 state_dict + ts/pooled 常量 + 配置清单。"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import torch
from tinysr_runtime import build_model

DEV, DT = torch.device("cuda"), torch.float16
OUT = "/cache/zihaohe/csig_isr/submit_v8/model_dir"
os.makedirs(OUT, exist_ok=True)
core, rep = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth",
                        device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto")
core.eval()
print("load report:", {k: rep[k] for k in ("alpha_mode", "merged", "step") if k in rep}, flush=True)
assets = {
    "tr": {k: v.cpu() for k, v in core.tr.state_dict().items()},
    "vae": {k: v.cpu() for k, v in core.vae.state_dict().items()},
    "ts": core.ts.cpu(),
    "pooled": core.pooled.cpu(),
    "alpha_gate": (core.alpha_gate.state_dict() if core.alpha_gate is not None else None),
    "alpha_cfg": ({"min": core.alpha_gate.alpha_min, "max": core.alpha_gate.alpha_max,
                   "hidden": core.alpha_gate.fc1.out_features}
                  if core.alpha_gate is not None else None),
    "scaling_factor": float(core.vae.config.scaling_factor),
}
torch.save(assets, os.path.join(OUT, "tinysr_v8_assets.pt"))
sz = os.path.getsize(os.path.join(OUT, "tinysr_v8_assets.pt")) / 1e6
print(f"assets 保存完成 {sz:.0f} MB", flush=True)
# tr/vae 的 config(from_config 重建用)
core.tr.config  # diffusers FrozenDict
json.dump(dict(core.tr.config), open(os.path.join(OUT, "tr_config.json"), "w"), default=str)
json.dump(dict(core.vae.config), open(os.path.join(OUT, "vae_config.json"), "w"), default=str)
print("EXPORT_DONE", flush=True)
