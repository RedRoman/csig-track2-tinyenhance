"""鱼鳞斑敏感性量化：把 distill_v2b 输出的平坦区鳞片抹掉，看六项掉多少。

方法：对整图 sr 做引导滤波式的边缘保持平滑（用 cv2.bilateralFilter 近似），
只在「LQ 低梯度区」（本该平坦的地方）替换为平滑版，边缘区保留原输出。
若 Score 几乎不变 => 定量证实评分对鳞片不敏感，且得到免费的决赛版清洗。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F, cv2, random
from PIL import Image
from settings import VAL
from metrics_utils import create_metrics
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
CK = f"{R}/train_out/distill_v2b/best.pth"
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def tiled(model, x, tile=512, ov=320):
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


def descale(sr01, lq01, flat_q=0.45, blend_soft=0.15):
    """sr/lq: [1,3,H,W] float。平坦掩码来自 LQ 梯度（不是 SR！SR 的鳞片自己就有梯度）。"""
    lq_np = (lq01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    sr_np = (sr01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(lq_np, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.abs(gx) + np.abs(gy), (0, 0), 8)
    thr = np.quantile(grad, flat_q)
    mask = np.clip((thr * (1 + blend_soft) - grad) / (thr * blend_soft * 2 + 1e-6), 0, 1)
    smooth = cv2.bilateralFilter(sr_np, d=9, sigmaColor=45, sigmaSpace=9)
    out = sr_np.astype(np.float32) * (1 - mask[..., None]) + smooth.astype(np.float32) * mask[..., None]
    flat_frac = float((mask > 0.5).mean())
    return torch.from_numpy(out / 255.).permute(2, 0, 1)[None].to(DEV), flat_frac


M = create_metrics(DEV)
model, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=False,
                       alpha_mode="fixed", fixed_alpha=1.0)
model.eval()

for tag_fn in [("原输出", None), ("平坦区去鳞", descale)]:
    tag, fn = tag_fn
    seed_all()
    acc = {k: [] for k in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")}
    fr_tot = []
    for l, g in PAIRS:
        lq = load(os.path.join(VAL, l)); gt = load(os.path.join(VAL, g)).to(DEV)
        sr = tiled(model, lq)
        if fn is not None:
            sr, ff = fn(sr, lq)
            fr_tot.append(ff)
        for k in acc:
            acc[k].append(float(M[k](sr, gt) if k in ("lpips", "dists") else M[k](sr)))
        del sr, gt; torch.cuda.empty_cache()
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    s = ((1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"] + m["maniqa"]
         + m["musiq"] / 100.0 + max(0.0, (10 - m["niqe"]) / 10.0))
    extra = f"  平坦区占比={np.mean(fr_tot):.2f}" if fr_tot else ""
    print(f"{tag:<10} lpips={m['lpips']:.4f} dists={m['dists']:.4f} niqe={m['niqe']:.3f} "
          f"musiq={m['musiq']:.2f} maniqa={m['maniqa']:.4f} clipiqa={m['clipiqa']:.4f} "
          f"Score={s:.4f}{extra}", flush=True)
print("\n解读：Score 掉幅 < 0.03 => 评分对鳞片不敏感成立，且此后处理可用于决赛清洗版")
