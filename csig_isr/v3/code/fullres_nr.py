"""在【完整 4K 图】上量无参考指标，与 512 中心 crop 对比。

动机：本地全部评测（训练验证、score_eval、alpha/soup 扫描、甚至 infer_4k 的
"4K NR summary"）都在 512 crop 上做。官方评的是 100 张完整 4K 图。
无参考指标对尺度极敏感，这可能解释为什么本地 +0.445 只换来榜分 +1.06。
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, pyiqa
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

DEV = torch.device("cuda")
DIRS = [("run8 已提交(71.54)",   "/cache/zihaohe/csig_isr/submit/output_dir"),
        ("combo_hf1500(72.60)", "/cache/zihaohe/csig_isr/submit_v3/output_dir")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DIRS[0][1], "*.jpg")))[:N]
M = {k: pyiqa.create_metric(k, device=DEV) for k in ("niqe", "musiq", "maniqa-pipal", "clipiqa")}

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]

res = {}
for lab, d in DIRS:
    acc = {f"{k}_{m}": [] for k in M for m in ("full","crop")}
    for i, nm in enumerate(names):
        p = os.path.join(d, nm)
        if not os.path.exists(p): continue
        t = load(p)
        _,_,H,W = t.shape
        crop = t[:,:,H//2-256:H//2+256, W//2-256:W//2+256]
        for k, fn in M.items():
            with torch.no_grad():
                for tag, x in (("full", t), ("crop", crop)):
                    try: acc[f"{k}_{tag}"].append(float(fn(x.to(DEV))))
                    except Exception as e:
                        if tag == "full" and i == 0: print(f"  {k} 全图失败: {type(e).__name__}")
        torch.cuda.empty_cache()
        if (i+1) % 5 == 0: print(f"  {lab} {i+1}/{len(names)}", flush=True)
    res[lab] = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}

print()
print(f"{'指标':<16}{'尺度':<7}" + ''.join(f'{l:>24}' for l,_ in DIRS) + f"{'差值':>12}")
print("-"*90)
for k in M:
    for tag in ("crop","full"):
        vs = [res[l].get(f"{k}_{tag}") for l,_ in DIRS]
        if any(v is None for v in vs): continue
        print(f"{k:<16}{tag:<7}" + ''.join(f'{v:>24.4f}' for v in vs) + f"{vs[1]-vs[0]:>+12.4f}")

def contrib(r, tag):
    try:
        return (r[f"clipiqa_{tag}"] + r[f"maniqa-pipal_{tag}"] + r[f"musiq_{tag}"]/100
                + max(0,(10-r[f"niqe_{tag}"])/10))
    except (KeyError, TypeError): return None
print()
for tag in ("crop","full"):
    cs = [contrib(res[l], tag) for l,_ in DIRS]
    if any(c is None for c in cs): continue
    print(f"四项无参考指标之和（{tag:>4}）: {cs[0]:.4f} -> {cs[1]:.4f}   增益 {cs[1]-cs[0]:+.4f}")
json.dump(res, open("/cache/zihaohe/csig_isr/eval_out/fullres_nr.json","w"), indent=2, ensure_ascii=False)
