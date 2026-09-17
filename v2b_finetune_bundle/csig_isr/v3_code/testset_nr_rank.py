"""在【真正的 100 张测试图】上、原生 4K 分辨率算四项无参考指标。

动机：全尺度 Score 只在 3 张验证图上测，而选 checkpoint 又是在这 3 张上从
几十个候选里取最大值 —— 选择性过拟合。且域诊断显示验证集与测试集统计差异极大
（lapvar 19.2 vs 71.8，差 3.7 倍）。测试集没有 GT，LPIPS/DISTS 测不了，但
NIQE/MUSIQ/MANIQA/CLIPIQA 不需要 GT，可直接在官方评测的同一批图上算。

判据：测试集排序是否与验证集一致。曾抓到 MUSIQ 在两者间符号翻转，
以及「测试集 NR 更高但榜分更低」（因 FR 差）。

用法: python testset_nr_rank.py [N] --dir 名称=路径 [--dir ...]
"""
import os, sys, glob, json, argparse, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
import numpy as np, torch
from PIL import Image
from metrics_utils import create_metrics
Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")


def seed_all(seed=1234):
    import random
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=100)
    ap.add_argument("--dir", action="append", required=True, help="名称=出图目录，可重复")
    ap.add_argument("--out_json", default="/cache/zihaohe/csig_isr/eval_out/testset_nr_rank.json")
    a = ap.parse_args()
    dirs = [tuple(d.split("=", 1)) for d in a.dir]

    M0 = create_metrics(DEV)
    M = {k: M0[k] for k in ("niqe", "musiq", "maniqa", "clipiqa")}
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dirs[0][1], "*.jpg")))[:a.n]
    print(f"基准目录 {dirs[0][0]} 有 {len(names)} 张\n", flush=True)

    res = {}
    for lab, d in dirs:
        have = [n for n in names if os.path.exists(os.path.join(d, n))]
        if not have:
            print(f"{lab}: 无出图，跳过"); continue
        seed_all(); acc = {k: [] for k in M}
        for i, nm in enumerate(have):
            t = load(os.path.join(d, nm)).to(DEV)
            with torch.no_grad():
                for k, f in M.items(): acc[k].append(float(f(t)))
            del t; torch.cuda.empty_cache()
            if (i+1) % 20 == 0: print(f"  {lab} {i+1}/{len(have)}", flush=True)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        m["nr_sum"] = (m["clipiqa"] + m["maniqa"] + m["musiq"]/100
                       + max(0, (10-m["niqe"])/10))
        m["n"] = len(have); res[lab] = m
        print(f"{lab}: nr_sum={m['nr_sum']:.4f}", flush=True)

    print()
    print("候选".ljust(24) + "".join(f"{k:>9}" for k in ("niqe","musiq","maniqa","clipiqa")) + f"{'NR四项和':>11}{'张数':>6}")
    print("-"*78)
    for lab, m in res.items():
        print(lab.ljust(24) + f"{m['niqe']:>9.3f}{m['musiq']:>9.2f}{m['maniqa']:>9.4f}"
              f"{m['clipiqa']:>9.4f}{m['nr_sum']:>11.4f}{m['n']:>6}")
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(res, open(a.out_json, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
