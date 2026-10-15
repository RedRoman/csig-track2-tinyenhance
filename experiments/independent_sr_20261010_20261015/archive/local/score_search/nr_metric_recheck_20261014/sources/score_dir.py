"""给【已经渲染好的输出图目录】打分。

  测试集口径(官方 100 张,无 GT → 只有 NR 四项):
      python score_dir.py --test_dir 名=目录 [--test_dir 名2=目录2] [--n 100]
      目录里须有 case1.jpg ... case100.jpg(或 .png),与 data/测试集 同名。
  验证集口径(3 组有 GT → 六项 + Score):
      python score_dir.py --val_dir 名=目录
      目录里须有 case1.* case2.* case3.*(模型对 case{i}_lq.jpg 的输出,尺寸须与 GT 一致)。
  两者可同时给。--out 写 json,--csv 另存逐图 csv。

输出解读:榜分 ∝ 2^(NR_test + FR_val)。选模型看 test NR(100 张)优先,val 三张的 Score 会说反话。
"""
import os, sys, glob, json, csv, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csig_metrics as CM


def _find(d, stem):
    for e in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        p = os.path.join(d, stem + e)
        if os.path.exists(p):
            return p
    return None


def score_test_dir(M, d, n=100, device="cuda", csv_path=None, log=print):
    import torch
    files = sorted(glob.glob(os.path.join(CM.TEST_DIR, "*.jpg")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))[:n]
    CM.seed_all()
    rows, per = [], []
    for i, f in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(f))[0]
        p = _find(d, stem)
        if p is None:
            raise FileNotFoundError(f"{d} 缺 {stem}.*")
        sr = CM.load_img(p, device)
        m = CM.measure(M, sr)
        rows.append(m); per.append({"file": os.path.basename(p), **m, "nr": CM.nr_contrib(m)})
        del sr; torch.cuda.empty_cache()
        if i % 25 == 0:
            log(f"    {i}/{len(files)}")
    mm = CM.mean_metrics(rows); mm["nr"] = CM.nr_contrib(mm); mm["n"] = len(rows)
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per[0].keys())); w.writeheader(); w.writerows(per)
    return mm


def score_val_dir(M, d, device="cuda", log=print):
    import torch
    CM.seed_all()
    rows = []
    for i, (lq, gt) in enumerate(CM.VAL_PAIRS, 1):
        p = _find(d, f"case{i}")
        if p is None:
            raise FileNotFoundError(f"{d} 缺 case{i}.*")
        sr = CM.load_img(p, device); g = CM.load_img(os.path.join(CM.VAL_DIR, gt), device)
        if sr.shape != g.shape:
            raise ValueError(f"case{i}: 输出 {tuple(sr.shape[-2:])} != GT {tuple(g.shape[-2:])}")
        rows.append(CM.measure(M, sr, g))
        del sr, g; torch.cuda.empty_cache()
    mm = CM.mean_metrics(rows)
    mm["fr"] = CM.fr_contrib(mm); mm["nr_val"] = CM.nr_contrib(mm); mm["score"] = CM.proxy_score(mm)
    return mm


def fmt_test(t):
    return (f"[测试集NR n={t['n']}] nr={t['nr']:.4f}  niqe={t['niqe']:.3f} musiq={t['musiq']:.2f} "
            f"maniqa={t['maniqa']:.4f} clipiqa={t['clipiqa']:.4f}")


def fmt_val(v):
    s = (f"[验证集全尺度] Score={v['score']:.4f}  FR={v['fr']:.4f} (lpips={v['lpips']:.4f} dists={v['dists']:.4f})  "
         f"NR_val={v['nr_val']:.4f} (niqe={v['niqe']:.3f} musiq={v['musiq']:.2f} maniqa={v['maniqa']:.4f} clipiqa={v['clipiqa']:.4f})")
    if "psnr" in v:
        s += f"  psnr={v['psnr']:.2f}"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_dir", action="append", default=[], help="名=目录(100 张测试集输出)")
    ap.add_argument("--val_dir", action="append", default=[], help="名=目录(case1/2/3 输出)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="", help="结果 json")
    ap.add_argument("--csv", action="store_true", help="测试集逐图 csv 存到目录旁 <名>_test_nr.csv")
    ap.add_argument("--psnr", action="store_true")
    a = ap.parse_args()
    if not a.test_dir and not a.val_dir:
        ap.error("至少给一个 --test_dir 或 --val_dir")
    import torch
    dev = torch.device("cuda")
    M = CM.create_metrics(dev, include_psnr=a.psnr)
    res = {}
    for spec in a.val_dir:
        name, d = spec.split("=", 1)
        v = score_val_dir(M, d, dev); res.setdefault(name, {})["val"] = v
        print(f"{name:14s} {fmt_val(v)}", flush=True)
    for spec in a.test_dir:
        name, d = spec.split("=", 1)
        t = score_test_dir(M, d, a.n, dev, csv_path=(f"{name}_test_nr.csv" if a.csv else None))
        res.setdefault(name, {})["test"] = t
        print(f"{name:14s} {fmt_test(t)}", flush=True)
    for name, r in res.items():
        if "val" in r and "test" in r:
            S = r["test"]["nr"] + r["val"]["fr"]
            r["S_proxy"] = S
            print(f"{name:14s} 代理 S = test NR {r['test']['nr']:.4f} + val FR {r['val']['fr']:.4f} = {S:.4f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
        print("saved", a.out)


if __name__ == "__main__":
    main()
