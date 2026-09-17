"""把 run12 的 separable VAE 换成官方 taesd3。

两者 latent 相关 0.96、尺度差约 2×，所以先用最小二乘拟合逐通道仿射
  z_off ~= a_c * z_sep + b_c
再把 DiT 输出的 latent 经此映射送进官方解码器。

零样本大概率不如原状（DiT 学会了预补偿坏解码器），关键看差多少：
差得不多 => 短微调就能吃下这 18dB；差很多 => 需要完整重训。
"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random
from PIL import Image
from settings import VAL, TS
from metrics_utils import create_metrics
from tinysr_runtime import build_model
sys.path.insert(0, TS)
from diffusers import AutoencoderTiny as TinyHF

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
R = "/cache/zihaohe/csig_isr"
OFF = glob.glob("/cache/zihaohe/hf/hub/models--madebyollin--taesd3/snapshots/*")[0]
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]
M = create_metrics(DEV)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


model, _ = build_model(f"{R}/train_out/run12_iqa30/best.pth", device=DEV, dtype=DT,
                       merge_lora=False, alpha_mode="fixed", fixed_alpha=1.0)
model.eval()
off = TinyHF.from_pretrained(OFF, torch_dtype=DT).to(DEV, dtype=DT).eval()
sep = model.vae

# ---- 拟合 z_off = a_c * z_sep + b_c（逐通道）----
zs_all, zo_all = [], []
with torch.no_grad():
    for l, g in PAIRS:
        for nm in (l, g):
            im = load(os.path.join(VAL, nm)); H, W = im.shape[-2:]
            for (y, x0) in [(H // 4, W // 4), (H // 2, W // 2), (3 * H // 4 - 512, 3 * W // 4 - 512)]:
                y = max(0, min(y, H - 512)); x0 = max(0, min(x0, W - 512))
                c = im[:, :, y:y + 512, x0:x0 + 512].to(DEV, dtype=DT) * 2 - 1
                zs_all.append(sep.encode(c).latents.float())
                zo_all.append(off.encode(c).latents.float())
zs = torch.cat([t.permute(1, 0, 2, 3).reshape(16, -1) for t in zs_all], 1)
zo = torch.cat([t.permute(1, 0, 2, 3).reshape(16, -1) for t in zo_all], 1)
a = ((zs * zo).mean(1) - zs.mean(1) * zo.mean(1)) / ((zs * zs).mean(1) - zs.mean(1) ** 2)
b = zo.mean(1) - a * zs.mean(1)
r2 = [float(1 - ((zo[c] - (a[c] * zs[c] + b[c])) ** 2).mean() / zo[c].var()) for c in range(16)]
print("逐通道仿射 a =", np.round(a.cpu().numpy(), 3).tolist())
print("        R2 =", np.round(r2, 4).tolist())
print("平均 R2 = {:.4f}".format(float(np.mean(r2))), flush=True)
A = a.view(1, 16, 1, 1).to(DT); B = b.view(1, 16, 1, 1).to(DT)


@torch.no_grad()
def tiled(fwd, x, tile=512, ov=256):
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
                p = fwd(x[:, :, y:y + tile, xx:xx + tile].to(DEV, dtype=DT)).float()
            out[:, :, y:y + tile, xx:xx + tile] += p * wm
            acc[:, :, y:y + tile, xx:xx + tile] += wm
    return (out / acc.clamp_min(1e-8)).clamp(0, 1)


@torch.no_grad()
def zo_of(lq01):
    """复刻 _impl，但只到 latent"""
    x = lq01 * 2 - 1; sf = sep.config.scaling_factor
    z = sep.encode(x).latents * sf
    pred = model.tr(hidden_states=z, timestep=model.ts,
                    pooled_projections=model.pooled, return_dict=False)[0]
    if model.lrrb is not None:
        pred = pred + model.lrrb(torch.cat([z, pred], 1))
    al = model.fixed_alpha.to(dtype=pred.dtype).reshape(1, 1, 1, 1)
    return (z - al * pred) / sf


def fwd_orig(x):
    return model(x)


def fwd_swap(x):
    zl = zo_of(x)
    img = off.decode(A * zl + B, return_dict=False)[0]
    return (img * 0.5 + 0.5).clamp(0, 1)


def fwd_rt_off(x):
    """对照：官方 taesd3 自往返（不经 DiT），看这条路的保真上界"""
    z = off.encode(x * 2 - 1).latents
    return (off.decode(z, return_dict=False)[0] * 0.5 + 0.5).clamp(0, 1)


def score(fwd, tag):
    seed_all()
    acc = {k: [] for k in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")}
    ps = []
    for l, g in PAIRS:
        lq = load(os.path.join(VAL, l)); gt = load(os.path.join(VAL, g)).to(DEV)
        sr = tiled(fwd, lq)
        for k in acc:
            acc[k].append(float(M[k](sr, gt) if k in ("lpips", "dists") else M[k](sr)))
        ps.append(10 * np.log10(1.0 / max(float(((sr - gt) ** 2).mean()), 1e-12)))
        del sr, gt; torch.cuda.empty_cache()
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    m["psnr"] = float(np.mean(ps))
    m["score"] = ((1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"] + m["maniqa"]
                  + m["musiq"] / 100.0 + max(0.0, (10 - m["niqe"]) / 10.0))
    print("{:<24} lpips={:.4f} dists={:.4f} niqe={:.3f} musiq={:.2f} maniqa={:.4f} "
          "clipiqa={:.4f} psnr={:.2f} score={:.4f}".format(
              tag, m["lpips"], m["dists"], m["niqe"], m["musiq"], m["maniqa"],
              m["clipiqa"], m["psnr"], m["score"]), flush=True)
    return m


res = {}
res["run12 原状(separable)"] = score(fwd_orig, "run12 原状(separable)")
res["run12 换官方taesd3解码"] = score(fwd_swap, "run12 换官方taesd3解码")
res["官方taesd3 自往返LQ"] = score(fwd_rt_off, "官方taesd3 自往返LQ")
json.dump(res, open(f"{R}/eval_out/vae_swap.json", "w"), indent=2, ensure_ascii=False)
