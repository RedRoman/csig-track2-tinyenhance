"""直接给像素 UNet checkpoint(pix_model.py 的 PixUNet/PixUNet2 权重)打分,不经 runner。
    python eval_ckpt.py 名=ckpt.pth [名2=ckpt2.pth ...] [--n 100] [--out res.json]
与 eval_submission 的区别:模型直推(fp16 autocast),不量时延,不落盘图片,快 2-3 倍;用于训练中选 checkpoint。"""
import os, sys, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import csig_metrics as CM


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("specs", nargs="+", help="名=ckpt.pth")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torch
    from pix_model import build_pix
    dev, DT = torch.device("cuda"), torch.float16
    M = CM.create_metrics(dev, include_psnr=True)

    @torch.no_grad()
    def infer(model, x01):
        with torch.autocast("cuda", dtype=DT):
            return model(x01.to(dev, dtype=DT)).float().clamp(0, 1)

    files = sorted(glob.glob(os.path.join(CM.TEST_DIR, "*.jpg")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))[:a.n]
    res = {}
    for spec in a.specs:
        name, ck = spec.split("=", 1)
        model = build_pix(ck, device=dev, dtype=DT); CM.seed_all()
        rows = []
        for lq, gt in CM.VAL_PAIRS:
            sr = infer(model, CM.load_img(os.path.join(CM.VAL_DIR, lq))); g = CM.load_img(os.path.join(CM.VAL_DIR, gt), dev)
            rows.append(CM.measure(M, sr, g)); del sr, g; torch.cuda.empty_cache()
        v = CM.mean_metrics(rows); v["fr"] = CM.fr_contrib(v); v["score"] = CM.proxy_score(v)
        print(f"{name:12s} [验证集全尺度] lpips={v['lpips']:.4f} dists={v['dists']:.4f} niqe={v['niqe']:.3f} musiq={v['musiq']:.2f} "
              f"maniqa={v['maniqa']:.4f} clipiqa={v['clipiqa']:.4f} psnr={v['psnr']:.2f} Score={v['score']:.4f} FR={v['fr']:.4f}", flush=True)
        rows = []
        for i, f in enumerate(files, 1):
            sr = infer(model, CM.load_img(f)); rows.append(CM.measure(M, sr)); del sr; torch.cuda.empty_cache()
            if i % 25 == 0:
                print(f"    {name} 测试集 {i}/{len(files)}", flush=True)
        t = CM.mean_metrics(rows); t["nr"] = CM.nr_contrib(t)
        print(f"{name:12s} [测试集NR] nr={t['nr']:.4f} niqe={t['niqe']:.3f} musiq={t['musiq']:.2f} maniqa={t['maniqa']:.4f} clipiqa={t['clipiqa']:.4f}"
              f"   代理 S={t['nr']+v['fr']:.4f}", flush=True)
        res[name] = {"val": v, "test": t, "S_proxy": t["nr"] + v["fr"]}
        del model; torch.cuda.empty_cache()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False); print("saved", a.out)


if __name__ == "__main__":
    main()
