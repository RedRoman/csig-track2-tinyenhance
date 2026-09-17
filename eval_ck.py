"""对单个 checkpoint 同时给出两个判据：
  A. 3 张验证图、原生 4K 全尺度的六项 Score（有 GT）
  B. 100 张真实测试图、原生 4K 的四项无参考 NR 合计（无 GT，但它是唯一
     把三次已知提交都排对了序的判据）

用法: eval_ck.py 标签=路径 [标签=路径 ...] [--test N]
环境变量 CSIG_VAE_PATH / CSIG_LATENT_SCALE 会被 build_model 与 _impl 读取，
所以换过 VAE 的模型必须带着对应环境变量单独跑。
"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random
from PIL import Image
from settings import VAL, TEST
from metrics_utils import create_metrics
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NTEST = 100
if "--test" in sys.argv:
    NTEST = int(sys.argv[sys.argv.index("--test") + 1])


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def tiled(model, x, tile=512, ov=256):
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
            with torch.autocast(device_type="cuda", dtype=DT):
                p = model(x[:, :, y:y + tile, xx:xx + tile].to(DEV, dtype=DT)).float()
            out[:, :, y:y + tile, xx:xx + tile] += p * wm
            acc[:, :, y:y + tile, xx:xx + tile] += wm
    return (out / acc.clamp_min(1e-8)).clamp(0, 1)


M = create_metrics(DEV)
tfiles = sorted(glob.glob(os.path.join(TEST, "*.jpg")),
                key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))[:NTEST]
print("VAE={}  LATENT_SCALE={}  测试图 {} 张".format(
    os.environ.get("CSIG_VAE_PATH", "(出厂 separable)"),
    os.environ.get("CSIG_LATENT_SCALE", "1.0"), len(tfiles)), flush=True)

out = {}
for spec in args:
    tag, ck = spec.split("=", 1)
    if not os.path.exists(ck):
        print("  缺 {} -> {}".format(tag, ck), flush=True); continue
    model, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="auto", fixed_alpha=1.0)
    model.eval()

    # ---- A. 验证集全尺度六项 ----
    seed_all()
    acc = {k: [] for k in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")}
    ps = []
    for l, g in PAIRS:
        gt = load(os.path.join(VAL, g)).to(DEV)
        sr = tiled(model, load(os.path.join(VAL, l)))
        for k in acc:
            acc[k].append(float(M[k](sr, gt) if k in ("lpips", "dists") else M[k](sr)))
        ps.append(10 * np.log10(1.0 / max(float(((sr - gt) ** 2).mean()), 1e-12)))
        del sr, gt; torch.cuda.empty_cache()
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    m["psnr"] = float(np.mean(ps))
    m["score"] = ((1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"] + m["maniqa"]
                  + m["musiq"] / 100.0 + max(0.0, (10 - m["niqe"]) / 10.0))
    print("  {:<16} [验证集全尺度] lpips={:.4f} dists={:.4f} niqe={:.3f} musiq={:.2f} "
          "maniqa={:.4f} clipiqa={:.4f} psnr={:.2f} Score={:.4f}".format(
              tag, m["lpips"], m["dists"], m["niqe"], m["musiq"], m["maniqa"],
              m["clipiqa"], m["psnr"], m["score"]), flush=True)

    # ---- B. 测试集 NR ----
    seed_all()
    a = {k: [] for k in ("niqe", "musiq", "maniqa", "clipiqa")}
    for i, f in enumerate(tfiles):
        sr = tiled(model, load(f))
        for k in a: a[k].append(float(M[k](sr)))
        del sr; torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print("      {} 测试集 {}/{}".format(tag, i + 1, len(tfiles)), flush=True)
    t = {k: float(np.mean(v)) for k, v in a.items()}
    t["nr"] = (t["clipiqa"] + t["maniqa"] + t["musiq"] / 100.0
               + max(0.0, (10 - t["niqe"]) / 10.0))
    print("  {:<16} [测试集100张] niqe={:.3f} musiq={:.2f} maniqa={:.4f} clipiqa={:.4f} "
          "NR={:.4f}".format(tag, t["niqe"], t["musiq"], t["maniqa"], t["clipiqa"], t["nr"]),
          flush=True)

    out[tag] = {"val_fullres": m, "test_nr": t}
    del model; torch.cuda.empty_cache()
    json.dump(out, open("{}/eval_out/eval_ck_{}.json".format(R, args[0].split("=")[0]), "w"),
              indent=2, ensure_ascii=False)

print()
print("历史锚点：run8 NR 2.4562 -> 榜 71.54 | combo NR 2.6144 -> 榜 72.6047 | 学弟h75 NR 1.5302 -> 榜 25.3359")
print("现有模型：ref1 2.5986 | run12 2.5250 | run11 2.4903 | run8 2.4535 | LQ 1.1992")
