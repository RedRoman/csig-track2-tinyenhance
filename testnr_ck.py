"""直接从 checkpoint 在 100 张真实测试图上算四项无参考指标（原生 4K）。
这是唯一一个把三次已知提交排对了序的判据，重新用它给现有模型排队。"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random
from PIL import Image
from settings import TEST
from metrics_utils import create_metrics
from tinysr_runtime import build_model
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CKS = [("run8", f"{R}/train_out/tinysr8_fid/best.pth"),
       ("run11", f"{R}/train_out/run11_big_iqa20/best.pth"),
       ("run12", f"{R}/train_out/run12_iqa30/best.pth"),
       ("ref1",  f"{R}/train_out/ref1/best.pth")]


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


KEYS = ("niqe", "musiq", "maniqa", "clipiqa")


def summarize(a):
    m = {k: float(np.mean(v)) for k, v in a.items()}
    m["nr"] = (m["clipiqa"] + m["maniqa"] + m["musiq"] / 100.0
               + max(0.0, (10 - m["niqe"]) / 10.0))
    return m


def show(tag, m):
    print("  {:<7} niqe={:.3f} musiq={:.2f} maniqa={:.4f} clipiqa={:.4f}  NR={:.4f}".format(
        tag, m["niqe"], m["musiq"], m["maniqa"], m["clipiqa"], m["nr"]), flush=True)


M = create_metrics(DEV)
files = sorted(glob.glob(os.path.join(TEST, "*.jpg")),
               key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))[:N]
print("测试图 {} 张".format(len(files)), flush=True)
res = {}

seed_all(); a = {k: [] for k in KEYS}
for f in files:
    x = load(f).to(DEV)
    for k in a: a[k].append(float(M[k](x)))
    del x; torch.cuda.empty_cache()
res["原始LQ"] = summarize(a); show("原始LQ", res["原始LQ"])

for tag, ck in CKS:
    if not os.path.exists(ck):
        print("  缺 {}".format(tag), flush=True); continue
    model, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=1.0)
    model.eval(); seed_all()
    a = {k: [] for k in KEYS}
    for i, f in enumerate(files):
        sr = tiled(model, load(f))
        for k in a: a[k].append(float(M[k](sr)))
        del sr; torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print("    {} {}/{}".format(tag, i + 1, len(files)), flush=True)
    res[tag] = summarize(a); show(tag, res[tag])
    del model; torch.cuda.empty_cache()
    json.dump(res, open(f"{R}/eval_out/testnr_ck.json", "w"), indent=2, ensure_ascii=False)

print()
print("{:<10}{:>8}{:>9}{:>9}{:>10}{:>10}".format("模型", "NIQE", "MUSIQ", "MANIQA", "CLIPIQA", "NR合计"))
for k, v in sorted(res.items(), key=lambda kv: -kv[1]["nr"]):
    print("{:<10}{:>8.3f}{:>9.2f}{:>9.4f}{:>10.4f}{:>10.4f}".format(
        k, v["niqe"], v["musiq"], v["maniqa"], v["clipiqa"], v["nr"]))
print()
print("历史锚点：run8 NR 2.4562 -> 榜 71.54 | combo NR 2.6144 -> 榜 72.6047 | 学弟h75 NR 1.5302 -> 榜 25.3359")
