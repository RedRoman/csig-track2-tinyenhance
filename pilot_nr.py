"""三教师 pilot 输出的 NR 四项——预判学生 NR 会被带向哪里。"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch
from PIL import Image
from metrics_utils import create_metrics

DEV = torch.device("cuda")
M = create_metrics(DEV)
D = "/cache/zihaohe/distill_data"

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV)

sr_map = {os.path.basename(f).split("_")[1]: f for f in glob.glob(f"{D}/pdit_pilot_sr/*_sr.png")}
arms = {
    "VOSR_v3_mixlq": lambda k: f"{D}/teacher_v3_mixlq/{k}.png",
    "PixelDiT": lambda k: sr_map[k],
    "VOSR2": lambda k: f"{D}/vosr2_pilot_sr/{k}.png",
    "GT": lambda k: f"{D}/pilot_gt/{k}.png",
}
for tag, fn in arms.items():
    vals = {m: [] for m in ("niqe", "musiq", "maniqa", "clipiqa")}
    for i in range(0, 200, 2):   # 100 张够稳
        x = load(fn(f"{i:06d}"))
        for m in vals: vals[m].append(float(M[m](x)))
    r = {m: float(np.mean(v)) for m, v in vals.items()}
    nr = (10 - r["niqe"]) / 10 + r["musiq"] / 100 + r["maniqa"] + r["clipiqa"]
    print(f"{tag:<14} niqe={r['niqe']:.3f} musiq={r['musiq']:.2f} maniqa={r['maniqa']:.4f} "
          f"clipiqa={r['clipiqa']:.4f} NR合计={nr:.4f}", flush=True)
