"""测「逐块 argmax 包络」比「最强单模型」高多少 —— Oracle Best-of-N 教师的上限。

训练数据有 GT，所以每张裁块的真实六项分可算。若逐块挑最好的那个模型输出，
得到的包络严格 >= 任何单模型。这个差值就是 Oracle 教师这条路的收益上限；
太小就不值得做。

MUSIQ 走 mosaic 口径（512 直接算与官方 4K 排序相关 -0.544，是反的）。

用法: envelope.py --family tinysr|vosr [--n 200]
两族分开跑（VOSR 和 TinySR 各有一个 models 包，同进程会撞名），结果落盘后合并。
"""
import os, sys, glob, json, argparse, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "20")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["tinysr", "vosr"], required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", default="/cache/zihaohe/csig_isr/eval_out")
args = ap.parse_args()

CODE = "/cache/zihaohe/csig_isr/v3/code"
sys.path.insert(0, CODE)
if args.family == "vosr":
    sys.path.insert(0, "/cache/zihaohe/VOSR")

import cv2
import numpy as np, torch, random
from PIL import Image
from settings import ROOT
from metrics_utils import create_metrics
from degradation import degrade as degrade_legacy

Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
KEYS = ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")

TINYSR = [("run8", "tinysr8_fid"), ("run12", "run12_iqa30"), ("ref1", "ref1"),
          ("sepctrl", "sep_control"), ("mosaic", "mosaic_musiq"),
          ("mosaic_cap", "mosaic_cap"), ("mosaic_cap2", "mosaic_cap2"),
          ("combo_exact", "combo_exact")]
VOSR = [("vosr_cap", "vosr_cap"), ("vosr_cap2", "vosr_cap2"), ("vosr_iqa1", "vosr_iqa1.0")]


def mosaic(t):
    H, W = 2304, 3072
    ry = -(-H // t.shape[-2]); rx = -(-W // t.shape[-1])
    return t.repeat(1, 1, ry, rx)[:, :, :H, :W]


def score6(M, sr, gt):
    v = {}
    for k in KEYS:
        if k in ("lpips", "dists"):
            v[k] = float(M[k](sr, gt))
        elif k == "musiq":
            v[k] = float(M[k](mosaic(sr)))
        else:
            v[k] = float(M[k](sr))
    v["score"] = ((1 - v["lpips"]) + (1 - v["dists"]) + v["clipiqa"] + v["maniqa"]
                  + v["musiq"] / 100.0 + max(0.0, (10 - v["niqe"]) / 10.0))
    return v


def make_crops(n, size=512, seed=20260819):
    paths = sorted(glob.glob(os.path.join(ROOT, "train_data", "lsdir512", "*.png")))
    rng0 = np.random.RandomState(seed)
    pick = rng0.choice(len(paths), size=min(n, len(paths)), replace=False)
    out = []
    for i in pick:
        rng = np.random.RandomState(int(i) * 7919 % (2 ** 31))
        gt = np.asarray(Image.open(paths[i]).convert("RGB"), dtype=np.uint8)
        h, w, _ = gt.shape
        if h > size or w > size:
            y = rng.randint(0, max(h - size, 0) + 1); x = rng.randint(0, max(w - size, 0) + 1)
            gt = gt[y:y + size, x:x + size]
        if gt.shape[0] != size or gt.shape[1] != size:
            gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_CUBIC)
        gt = np.ascontiguousarray(gt)
        lq = degrade_legacy(gt, rng)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.).permute(2, 0, 1)[None]
        out.append((to_t(lq), to_t(gt)))
    return out


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


M = create_metrics(DEV)
crops = make_crops(args.n)
print(f"裁块 {len(crops)} 张，family={args.family}", flush=True)

per_model = {}
if args.family == "tinysr":
    from tinysr_runtime import build_model
    cands = [(tag, f"{ROOT}/train_out/{d}/best.pth") for tag, d in TINYSR]
    for tag, ck in cands:
        if not os.path.exists(ck):
            print(f"  缺 {tag}", flush=True); continue
        m, _ = build_model(ck, device=DEV, dtype=DT, merge_lora=False,
                           alpha_mode="fixed", fixed_alpha=1.0)
        m.eval(); seed_all()
        rows = []
        for lq, gt in crops:
            with torch.no_grad(), torch.autocast("cuda", dtype=DT):
                sr = m(lq.to(DEV, dtype=DT)).float().clamp(0, 1)
            rows.append(score6(M, sr, gt.to(DEV)))
            del sr; torch.cuda.empty_cache()
        per_model[tag] = rows
        print(f"  {tag:<14} 均分 {np.mean([r['score'] for r in rows]):.4f}", flush=True)
        del m; torch.cuda.empty_cache()
else:
    src = open("/cache/zihaohe/train_vosr_csig.py").read().replace("\nmain()\n", "\n")
    ns = {"__name__": "tv_mod", "__file__": "/cache/zihaohe/train_vosr_csig.py"}
    exec(compile(src, "train_vosr_csig.py", "exec"), ns)
    for tag, d in VOSR:
        ck = f"{ROOT}/train_out/{d}/best.pth"
        if not os.path.exists(ck):
            print(f"  缺 {tag}", flush=True); continue
        sd = torch.load(ck, map_location="cpu"); a = sd["args"]
        m = ns["build"](a["ckpt"], bool(a.get("train_decoder")),
                        a.get("dino_layers", 9), a.get("dino_size", 0))
        m.tr, _ = ns["add_lora"](m.tr, a["lora_r"], a["lora_alpha"])
        m.tr = m.tr.to(DEV)
        if hasattr(m.tr, "base_model"):
            bm = m.tr.base_model.model
            if hasattr(bm, "forward_flexible"):
                bm.forward = bm.forward_flexible
        m.tr.load_state_dict(sd["lora"], strict=False)
        if sd.get("dec"):
            m.dec.load_state_dict(sd["dec"])
        for p in m.parameters():
            p.data = p.data.to(DT)
        m = m.eval(); seed_all()
        rows = []
        for lq, gt in crops:
            with torch.no_grad(), torch.autocast("cuda", dtype=DT):
                sr = m(lq.to(DEV, dtype=DT)).float().clamp(0, 1)
            rows.append(score6(M, sr, gt.to(DEV)))
            del sr; torch.cuda.empty_cache()
        per_model[tag] = rows
        print(f"  {tag:<14} 均分 {np.mean([r['score'] for r in rows]):.4f}", flush=True)
        del m; torch.cuda.empty_cache()

path = os.path.join(args.out, f"envelope_{args.family}.json")
json.dump(per_model, open(path, "w"), indent=2, ensure_ascii=False)
print(f"已存 {path}", flush=True)

# 若两族都在，合并算包络
other = os.path.join(args.out, f"envelope_{'vosr' if args.family=='tinysr' else 'tinysr'}.json")
allm = dict(per_model)
if os.path.exists(other):
    allm.update(json.load(open(other)))
tags = list(allm)
n = min(len(v) for v in allm.values())
best_single = max(tags, key=lambda t: np.mean([r["score"] for r in allm[t][:n]]))
bs = float(np.mean([r["score"] for r in allm[best_single][:n]]))
env = float(np.mean([max(allm[t][i]["score"] for t in tags) for i in range(n)]))
print()
print(f"参与模型 {len(tags)} 个：{', '.join(tags)}")
print(f"最强单模型  {best_single}  {bs:.4f}")
print(f"逐块 argmax 包络        {env:.4f}")
print(f"包络增益                +{env-bs:.4f}")
cnt = {}
for i in range(n):
    w = max(tags, key=lambda t: allm[t][i]["score"])
    cnt[w] = cnt.get(w, 0) + 1
print("每个模型当选次数：", {k: v for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])})
