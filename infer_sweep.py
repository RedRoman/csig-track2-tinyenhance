"""推理端免费增益扫描（零训练、p-不敏感）：wavelet colorfix × tile overlap。

排队很久的两个纯推理旋钮，从没在全尺度+koniq 口径上验证过：
  - colorfix：提交管线 infer_4k.py 默认带 wavelet colorfix，而我们所有选型评测
    (fullres_rank/eval_ck) 都没带 —— 口径不一致，方向未知，必须测。
  - overlap：一直固定 256，никогда没扫过。

对 distill_v2b 在 3 张验证图上跑 {none, wavelet} × {192, 256, 320}，六项全尺度。
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
from tinysr_runtime import build_model
from torchvision.transforms import ToPILImage
from utils.wavelet_color_fix import wavelet_color_fix

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
CK = sys.argv[1] if len(sys.argv) > 1 else f"{R}/train_out/distill_v2b/best.pth"
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def tiled(model, x, ov):
    tile = 512
    _, _, H, W = x.shape; st = tile - ov
    ys = list(range(0, max(H - tile, 0) + 1, st)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, st)) or [0]
    if ys[-1] + tile < H: ys.append(max(H - tile, 0))
    if xs[-1] + tile < W: xs.append(max(W - tile, 0))
    t = torch.linspace(-1, 1, tile, device=DEV); g = torch.exp(-(t ** 2) / (2 * 0.45 ** 2))
    wm = (g[:, None] * g[None, :])[None, None].float()
    out = torch.zeros(1, 3, H, W, device=DEV); acc = torch.zeros(1, 1, H, W, device=DEV)
    for y in ys:
        for xx in xs:
            with torch.autocast("cuda", dtype=DT):
                p = model(x[:, :, y:y + tile, xx:xx + tile].to(DEV, dtype=DT)).float()
            out[:, :, y:y + tile, xx:xx + tile] += p * wm
            acc[:, :, y:y + tile, xx:xx + tile] += wm
    return (out / acc.clamp_min(1e-8)).clamp(0, 1)


M = create_metrics(DEV)
model, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=False,
                       alpha_mode="fixed", fixed_alpha=1.0)
model.eval()
res = {}
for ov in (192, 256, 320):
    for cf in ("none", "wavelet"):
        seed_all()
        acc = {k: [] for k in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")}
        for l, g in PAIRS:
            lq = load(os.path.join(VAL, l)); gt = load(os.path.join(VAL, g)).to(DEV)
            sr = tiled(model, lq, ov)
            if cf == "wavelet":
                fixed = wavelet_color_fix(ToPILImage()(sr[0].cpu()), ToPILImage()(lq[0]))
                sr = torch.from_numpy(np.asarray(fixed, np.float32) / 255.
                                      ).permute(2, 0, 1)[None].to(DEV)
            for k in acc:
                acc[k].append(float(M[k](sr, gt) if k in ("lpips", "dists") else M[k](sr)))
            del sr, gt; torch.cuda.empty_cache()
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        s = ((1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"] + m["maniqa"]
             + m["musiq"] / 100.0 + max(0.0, (10 - m["niqe"]) / 10.0))
        res[f"ov{ov}_{cf}"] = {**m, "score": s}
        print(f"ov={ov} colorfix={cf:<8} lpips={m['lpips']:.4f} dists={m['dists']:.4f} "
              f"niqe={m['niqe']:.3f} musiq={m['musiq']:.2f} maniqa={m['maniqa']:.4f} "
              f"clipiqa={m['clipiqa']:.4f} Score={s:.4f}", flush=True)
json.dump(res, open(f"{R}/eval_out/infer_sweep.json", "w"), indent=2, ensure_ascii=False)
print("\n基线对照：distill_v2b ov256/none 此前实测 Score=4.0522")
