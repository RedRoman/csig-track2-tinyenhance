"""能不能在 512 训练块上构造出「4K 口径」的 MUSIQ 代理？

已测定：MUSIQ 的响应依赖图像的绝对尺寸。同一份内容，
  原生 4K 排序   run12 > run8 > ref1 > sepctrl
  直接 512 排序  sepctrl > ref1 > run12 > run8   （整个颠倒）
连 /2、/4 都还在错误的一侧，只有满 4K 尺寸才给出正确排序。

若把 512 输出平铺成 4096x3072 的画布（8x6=48 份），MUSIQ 就看到一张 4K 尺寸、
纹理统计与我们输出一致的图。若它的排序与真实 4K 一致，就得到一个可导、
且尺度正确的 MUSIQ 训练代理。

同时测几种候选口径，挑排序对得上的：
  crop512      直接在 512 块上算（现状，已知错）
  mosaic4K     512 块平铺成 4096x3072
  mosaic2K     512 块平铺成 2048x1536
  pad4K        512 块居中、四周补反射到 4096x3072
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, torch.nn.functional as F, random
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
# 真实 4K 口径的排序（A1 实测），代理必须复现它
TRUTH = {"run12": 55.135, "run8": 52.227, "ref1": 41.835, "sepctrl": 37.207}


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


M = create_metrics(DEV)
musiq = lambda t: float(M["musiq"](t))


def mosaic(t, H, W):
    ry, rx = int(np.ceil(H / t.shape[-2])), int(np.ceil(W / t.shape[-1]))
    return t.repeat(1, 1, ry, rx)[:, :, :H, :W].contiguous()


def pad_reflect(t, H, W):
    _, _, h, w = t.shape
    top, left = (H - h) // 2, (W - w) // 2
    out = t
    # 反射填充一次最多补一个边长，分多次补
    while out.shape[-2] < H or out.shape[-1] < W:
        ph = min(out.shape[-2] - 1, (H - out.shape[-2] + 1) // 2)
        pw = min(out.shape[-1] - 1, (W - out.shape[-1] + 1) // 2)
        out = F.pad(out, (pw, pw, ph, ph), mode="reflect")
    _, _, h2, w2 = out.shape
    y0, x0 = (h2 - H) // 2, (w2 - W) // 2
    return out[:, :, y0:y0 + H, x0:x0 + W].contiguous()


PROTOS = {
    "crop512":  lambda t: t,
    "mosaic4K": lambda t: mosaic(t, 3072, 4096),
    "mosaic2K": lambda t: mosaic(t, 1536, 2048),
    "pad4K":    lambda t: pad_reflect(t, 3072, 4096),
}

imgs = [load(os.path.join(VAL, n)) for n in LQS]
res = {}
for tag, ck in CKS:
    if not os.path.exists(ck):
        print("  缺 {}".format(tag), flush=True); continue
    model, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=1.0)
    model.eval(); seed_all()
    acc = {k: [] for k in PROTOS}
    for lq in imgs:
        H, W = lq.shape[-2:]
        c = lq[:, :, H // 2 - 256:H // 2 + 256, W // 2 - 256:W // 2 + 256].to(DEV, dtype=DT)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DT):
            o = model(c).float().clamp(0, 1)
        for k, fn in PROTOS.items():
            acc[k].append(musiq(fn(o)))
        del o; torch.cuda.empty_cache()
    res[tag] = {k: float(np.mean(v)) for k, v in acc.items()}
    print("  {:<8} ".format(tag) + "  ".join(
        "{}={:.2f}".format(k, res[tag][k]) for k in PROTOS), flush=True)
    del model; torch.cuda.empty_cache()

mods = list(res)
truth_rank = sorted(mods, key=lambda m: -TRUTH[m])
print()
print("真实 4K 排序      " + " > ".join(truth_rank) +
      "   ({})".format(", ".join("{}={:.1f}".format(m, TRUTH[m]) for m in truth_rank)))
for k in PROTOS:
    rk = sorted(mods, key=lambda m: -res[m][k])
    tv = np.array([TRUTH[m] for m in mods]); pv = np.array([res[m][k] for m in mods])
    r = float(np.corrcoef(tv, pv)[0, 1]) if len(mods) > 2 else float("nan")
    print("{:<12} {:<42} r={:+.3f}  {}".format(
        k, " > ".join(rk), r, "<<< 排序一致" if rk == truth_rank else "排序不符"))
json.dump(res, open(f"{R}/eval_out/musiq_proxy.json", "w"), indent=2, ensure_ascii=False)
