"""找出能复现真实 4K 排序的最小 mosaic 画布尺寸。

已知：2048x1536 时 r=-0.634（还在错误一侧），4096x3072 时 r=+0.997。
阈值在中间。画布越小，训练时 MUSIQ 的显存和算力越省。
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
LQS = ["case1_lq.jpg", "case2_lq.jpg", "case3_lq.jpg"]
CKS = [("run8", f"{R}/train_out/tinysr8_fid/best.pth"),
       ("run12", f"{R}/train_out/run12_iqa30/best.pth"),
       ("ref1", f"{R}/train_out/ref1/best.pth"),
       ("sepctrl", f"{R}/train_out/sep_control/best.pth")]
TRUTH = {"run12": 55.135, "run8": 52.227, "ref1": 41.835, "sepctrl": 37.207}
SIZES = [(1536, 2048), (1792, 2304), (2048, 2560), (2304, 3072),
         (2560, 3328), (2816, 3584), (3072, 4096)]


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


M = create_metrics(DEV)


def mosaic(t, H, W):
    ry = -(-H // t.shape[-2]); rx = -(-W // t.shape[-1])
    return t.repeat(1, 1, ry, rx)[:, :, :H, :W].contiguous()


imgs = [load(os.path.join(VAL, n)) for n in LQS]
res = {}
for tag, ck in CKS:
    model, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=1.0)
    model.eval(); seed_all()
    acc = {s: [] for s in SIZES}
    for lq in imgs:
        H, W = lq.shape[-2:]
        c = lq[:, :, H // 2 - 256:H // 2 + 256, W // 2 - 256:W // 2 + 256].to(DEV, dtype=DT)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DT):
            o = model(c).float().clamp(0, 1)
        for s in SIZES:
            acc[s].append(float(M["musiq"](mosaic(o, s[0], s[1]))))
        del o; torch.cuda.empty_cache()
    res[tag] = {str(s): float(np.mean(v)) for s, v in acc.items()}
    print("  {} 完成".format(tag), flush=True)
    del model; torch.cuda.empty_cache()

mods = list(res)
truth_rank = sorted(mods, key=lambda m: -TRUTH[m])
tv = np.array([TRUTH[m] for m in mods])
print()
print("真实 4K 排序  " + " > ".join(truth_rank))
print("{:<14}{:>9}{:>44}{:>10}".format("画布", "像素(M)", "排序", "相关r"))
for s in SIZES:
    pv = np.array([res[m][str(s)] for m in mods])
    rk = sorted(mods, key=lambda m: -res[m][str(s)])
    r = float(np.corrcoef(tv, pv)[0, 1])
    print("{:<14}{:>9.2f}{:>44}{:>10.3f}  {}".format(
        "{}x{}".format(s[1], s[0]), s[0] * s[1] / 1e6, " > ".join(rk), r,
        "<<< 一致" if rk == truth_rank else ""))
json.dump(res, open(f"{R}/eval_out/musiq_size_scan.json", "w"), indent=2, ensure_ascii=False)
