"""weight soup：distill_v6 × distill_v6gan，扫混合比例。

两臂优势正交（v6: MANIQA/MUSIQ 高；v6gan: NIQE/FR 好），
同族同初始化（都从 v5a 出发），线性插值合法性强。
LoRA 权重逐张量插值，crop 快评挑最优比例，再全尺度精评。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "20")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import torch
from tinysr_runtime import build_model, checkpoint_payload
from metrics_utils import create_metrics
from eval_utils import collect_crops, evaluate_model

DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
A = torch.load(f"{R}/train_out/distill_v6/best.pth", map_location="cpu", weights_only=False)
B = torch.load(f"{R}/train_out/distill_v6gan/best.pth", map_location="cpu", weights_only=False)

M = create_metrics(DEV, as_loss=True, include_psnr=True)
val_items = collect_crops(12)

best_alpha, best_score = None, -1e9
for alpha in (0.3, 0.5, 0.7):
    mix = dict(A)
    mix["lora"] = {k: alpha * A["lora"][k].float() + (1 - alpha) * B["lora"][k].float()
                   for k in A["lora"]}
    tmp = f"{R}/train_out/soup_v6/alpha{alpha}.pth"
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    torch.save(mix, tmp)
    m, _ = build_model(tmp, device=DEV, dtype=DT, merge_lora=False,
                       alpha_mode="fixed", fixed_alpha=1.0)
    m.eval()
    ev = evaluate_model(m, val_items, M, device=DEV, dtype=DT, robust_beta=0.15)
    print(f"alpha={alpha} (v6 权重占比): crop robust={ev['robust_score']:.4f}", flush=True)
    if ev["robust_score"] > best_score:
        best_score, best_alpha = ev["robust_score"], alpha
    del m; torch.cuda.empty_cache()

print(f"\n最优 alpha={best_alpha} robust={best_score:.4f}", flush=True)
os.replace(f"{R}/train_out/soup_v6/alpha{best_alpha}.pth", f"{R}/train_out/soup_v6/best.pth")
print("已存 soup_v6/best.pth", flush=True)
