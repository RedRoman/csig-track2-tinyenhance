"""三方 weight soup：distill_v6 × distill_v6gan × distill_v7，扫 7 组凸组合。

v6(4.1705) 最优做主臂；v6gan(4.1436) NIQE/FR 好；v7(4.1678) 是 v6+GAN 续训。
同族同初始化（均出自 v5a 谱系），LoRA 逐张量凸组合。
crop 快评挑最优，胜者存 soup3/best.pth 再全尺度精评（eval_ck.py 另跑）。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "20")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import torch
from tinysr_runtime import build_model
from metrics_utils import create_metrics
from eval_utils import collect_crops, evaluate_model

DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
A = torch.load(f"{R}/train_out/distill_v6/best.pth", map_location="cpu", weights_only=False)
B = torch.load(f"{R}/train_out/distill_v6gan/best.pth", map_location="cpu", weights_only=False)
C = torch.load(f"{R}/train_out/distill_v7/best.pth", map_location="cpu", weights_only=False)

M = create_metrics(DEV, as_loss=True, include_psnr=True)
val_items = collect_crops(12)

COMBOS = [(0.7, 0.15, 0.15), (0.6, 0.2, 0.2), (0.5, 0.25, 0.25),
          (0.34, 0.33, 0.33), (0.5, 0.1, 0.4), (0.4, 0.2, 0.4), (0.5, 0.0, 0.5)]

os.makedirs(f"{R}/train_out/soup3", exist_ok=True)
best_key, best_score = None, -1e9
for wa, wb, wc in COMBOS:
    mix = dict(A)
    mix["lora"] = {k: (wa * A["lora"][k].float() + wb * B["lora"][k].float()
                       + wc * C["lora"][k].float()) for k in A["lora"]}
    tag = f"{wa}_{wb}_{wc}"
    tmp = f"{R}/train_out/soup3/w{tag}.pth"
    torch.save(mix, tmp)
    m, _ = build_model(tmp, device=DEV, dtype=DT, merge_lora=False,
                       alpha_mode="fixed", fixed_alpha=1.0)
    m.eval()
    ev = evaluate_model(m, val_items, M, device=DEV, dtype=DT, robust_beta=0.15)
    print(f"w=({wa},{wb},{wc}): crop robust={ev['robust_score']:.4f}", flush=True)
    if ev["robust_score"] > best_score:
        best_score, best_key = ev["robust_score"], tag
    del m; torch.cuda.empty_cache()

print(f"\n最优 w={best_key} robust={best_score:.4f}", flush=True)
os.replace(f"{R}/train_out/soup3/w{best_key}.pth", f"{R}/train_out/soup3/best.pth")
for f in os.listdir(f"{R}/train_out/soup3"):
    if f.startswith("w"): os.remove(f"{R}/train_out/soup3/{f}")
print("已存 soup3/best.pth", flush=True)
