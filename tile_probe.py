"""tiled 渲染损失探针：E1 相位对拍 + E2 oracle 对拍。ckpt=distill_v8。

E1: 网格平移 128px，同一像素两种身份（中心/接缝角），配对比 err-vs-GT。
E2: 512 单次前向的中心 256 = 无切块理想解，量 tiled 偏离理想解的绝对量。
颜色修复两边都不做（只测切块本身）。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from settings import VAL
from tinysr_runtime import build_model
from metrics_utils import create_metrics

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
CK = "/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth"
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]
TILE, ST = 512, 256

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]

@torch.no_grad()
def tiled(model, x):
    _, _, H, W = x.shape
    ys = list(range(0, max(H - TILE, 0) + 1, ST)) or [0]
    xs = list(range(0, max(W - TILE, 0) + 1, ST)) or [0]
    if ys[-1] + TILE < H: ys.append(max(H - TILE, 0))
    if xs[-1] + TILE < W: xs.append(max(W - TILE, 0))
    t = torch.linspace(-1, 1, TILE, device=DEV)
    g = torch.exp(-(t ** 2) / (2 * 0.45 ** 2))
    wm = (g[:, None] * g[None, :])[None, None].float()
    out = torch.zeros(1, 3, H, W, device=DEV); acc = torch.zeros(1, 1, H, W, device=DEV)
    for y in ys:
        for xx in xs:
            with torch.autocast("cuda", dtype=DT):
                p = model(x[:, :, y:y+TILE, xx:xx+TILE].to(DEV, dtype=DT)).float()
            out[:, :, y:y+TILE, xx:xx+TILE] += p * wm
            acc[:, :, y:y+TILE, xx:xx+TILE] += wm
    return (out / acc.clamp_min(1e-8)).clamp(0, 1)

model, rep = build_model(CK, device=DEV, dtype=DT, merge_lora=False, alpha_mode="auto")
model.eval()
print("alpha_mode:", rep.get("alpha_mode"), flush=True)
M = create_metrics(DEV)

def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))

agg = {"c_c": [], "c_s": [], "to_c": [], "to_s": [],
       "pt_c": [], "po_c": [], "pt_s": [], "po_s": [],
       "lt_s": [], "lo_s": []}
for lqf, gtf in PAIRS:
    lq = load(os.path.join(VAL, lqf)); gt = load(os.path.join(VAL, gtf)).to(DEV)
    _, _, H, W = lq.shape
    srA = tiled(model, lq)                                   # 相位 0
    pad = F.pad(lq, (128, 0, 128, 0), mode="reflect")
    srB = tiled(model, pad)[:, :, 128:, 128:]                # 相位 +128
    errA = (srA - gt).abs().mean(1)[0]; errB = (srB - gt).abs().mean(1)[0]
    # 掩码限规则内区（避开边缘特殊 tile）
    yy, xx = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    inner = (yy >= 512) & (yy < H - 768) & (xx >= 512) & (xx < W - 768)
    my = torch.minimum(yy % 256, 256 - yy % 256); mx = torch.minimum(xx % 256, 256 - xx % 256)
    cen = inner & (my <= 48) & (mx <= 48)      # 相位0中心 = 相位1接缝角
    print(f"{lqf}: 掩码像素 {int(cen.sum())}", flush=True)
    eC = float(errA[cen].mean()); eS = float(errB[cen].mean())
    agg["c_c"].append(eC); agg["c_s"].append(eS)
    # E2: oracle 窗口
    for role, coords in (("c", [(r, c) for r in (768, 1280) for c in (768, 1536, 2304)]),
                         ("s", [(r + 128, c + 128) for r in (768, 1280) for c in (768, 1536, 2304)])):
        for (r, c) in coords:
            if r + 384 > H or c + 384 > W: continue
            with torch.no_grad(), torch.autocast("cuda", dtype=DT):
                o = model(lq[:, :, r-256:r+256, c-256:c+256].to(DEV, dtype=DT)).float().clamp(0, 1)
            oc = o[:, :, 128:384, 128:384]
            tc = srA[:, :, r-128:r+128, c-128:c+128]
            gc = gt[:, :, r-128:r+128, c-128:c+128]
            agg[f"to_{role}"].append(float((tc - oc).abs().mean()))
            agg[f"pt_{role}"].append(psnr(tc, gc)); agg[f"po_{role}"].append(psnr(oc, gc))
            if role == "s":
                agg["lt_s"].append(float(M["lpips"](tc, gc))); agg["lo_s"].append(float(M["lpips"](oc, gc)))

m = {k: float(np.mean(v)) for k, v in agg.items() if v}
print("\n=== E1 相位对拍（同一批像素，中心身份 vs 接缝角身份）===")
print(f"L1 err 中心身份={m['c_c']:.5f}  接缝身份={m['c_s']:.5f}  相对差 {(m['c_s']/m['c_c']-1)*100:+.2f}%")
print("\n=== E2 oracle 对拍（tiled vs 512 单次前向理想解）===")
print(f"tile中心位:  |tiled-oracle| L1={m['to_c']:.5f}  PSNRvsGT tiled={m['pt_c']:.3f} oracle={m['po_c']:.3f}")
print(f"接缝角位:    |tiled-oracle| L1={m['to_s']:.5f}  PSNRvsGT tiled={m['pt_s']:.3f} oracle={m['po_s']:.3f}")
print(f"接缝角LPIPS: tiled={m['lt_s']:.4f} oracle={m['lo_s']:.4f}  (差={m['lt_s']-m['lo_s']:+.4f})")
print("PROBE_DONE")
