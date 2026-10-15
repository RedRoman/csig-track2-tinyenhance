"""端到端模拟官方评测:加载提交件 model_dir/runner.py → 量 512 单发时延 → 现跑验证集 3 张 + 测试集 100 张 → 六项/NR → 榜分预测。

    python eval_submission.py --model_dir path/to/model_dir [--out_dir render_xxx] [--n 100] [--label G4]
    python eval_submission.py --zip 赛题二_xxx.zip                         # 自动解包取 model_dir
    python eval_submission.py --model_dir ... --ref_ms 0.70                 # 同机 G4 参照换算时延口径(见 predict.py)
    python eval_submission.py --model_dir ... --skip_render --out_dir d      # 已渲染过,只打分

流程与官方一致的部分:Runner(model_dir) 构造;infer 输入 [1,3,H,W] fp16 [-1,1],4K 整图直接喂;输出 clamp 到 [-1,1]。
不确定的部分(官方未公开):时延的精确量法与固定开销;输出是否经 JPEG 再评。这里按 512×512 fp16 单发、CUDA event
中位数计;渲染默认存 PNG(无损)供打分,--jpg_quality 97 可模拟 output_dir 口径。
"""
import os, sys, time, json, glob, argparse, tempfile, zipfile, importlib.util, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import csig_metrics as CM
from score_dir import score_test_dir, score_val_dir, fmt_test, fmt_val
from predict import predict, ANCHOR


def load_runner(model_dir):
    """按官方方式加载:cwd 切到 model_dir、sys.path 追加,importlib 载入 runner.py,取 Runner 类。"""
    import torch
    model_dir = os.path.abspath(model_dir)
    old = os.getcwd(); os.chdir(model_dir); sys.path.append(model_dir)
    try:
        spec = importlib.util.spec_from_file_location("runner", os.path.join(model_dir, "runner.py"))
        mod = importlib.util.module_from_spec(spec); sys.modules["runner"] = mod; spec.loader.exec_module(mod)
        t0 = time.time(); r = mod.Runner(model_dir); torch.cuda.synchronize(); t_init = time.time() - t0
    finally:
        os.chdir(old)
    return r, t_init


def latency_512(r, n=200, warm=30):
    import torch
    x = (torch.rand(1, 3, 512, 512, device="cuda", dtype=torch.float16) * 2 - 1)
    with torch.no_grad():
        for _ in range(warm):
            r.infer(x)
        torch.cuda.synchronize()
        ev, wall = [], []
        for _ in range(n):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); a.record(); r.infer(x); b.record(); torch.cuda.synchronize()
            wall.append((time.perf_counter() - t0) * 1e3); ev.append(a.elapsed_time(b))
    return {"event_med": float(np.median(ev)), "event_p10": float(np.percentile(ev, 10)), "event_min": float(np.min(ev)),
            "wall_med": float(np.median(wall)), "wall_min": float(np.min(wall)), "n": n}


def render(r, out_dir, n=100, jpg_quality=0, log=print):
    import torch
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    vd, td = os.path.join(out_dir, "val"), os.path.join(out_dir, "test")
    os.makedirs(vd, exist_ok=True); os.makedirs(td, exist_ok=True)

    def run_one(src, dst_stem, ddir):
        x = CM.load_img(src, "cuda").half() * 2 - 1
        with torch.no_grad():
            y = r.infer(x).float().clamp(-1, 1)
        img = Image.fromarray(((y[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).round().astype(np.uint8))
        if jpg_quality > 0:
            img.save(os.path.join(ddir, dst_stem + ".jpg"), quality=jpg_quality, subsampling=0)
        else:
            img.save(os.path.join(ddir, dst_stem + ".png"), compress_level=1)
        del x, y; torch.cuda.empty_cache()

    t0 = time.time()
    for i, (lq, _) in enumerate(CM.VAL_PAIRS, 1):
        run_one(os.path.join(CM.VAL_DIR, lq), f"case{i}", vd)
    files = sorted(glob.glob(os.path.join(CM.TEST_DIR, "*.jpg")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))[:n]
    for i, f in enumerate(files, 1):
        run_one(f, os.path.splitext(os.path.basename(f))[0], td)
        if i % 25 == 0:
            log(f"    rendered {i}/{len(files)}  ({time.time()-t0:.0f}s)")
    return vd, td, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir"); ap.add_argument("--zip")
    ap.add_argument("--out_dir", default="", help="渲染输出目录(默认 render_<label>)")
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--jpg_quality", type=int, default=0, help=">0 则渲染存 JPEG(模拟 output_dir);默认 PNG 无损")
    ap.add_argument("--ref_ms", type=float, default=None, help="同机 examples/submit_pixG4_g 的 512 单发 ms(时延口径换算)")
    ap.add_argument("--skip_render", action="store_true"); ap.add_argument("--skip_latency", action="store_true")
    ap.add_argument("--out", default="", help="结果 json(默认 <out_dir>/result.json)")
    a = ap.parse_args()
    if not a.model_dir and not a.zip:
        ap.error("需要 --model_dir 或 --zip")
    if a.zip:
        tmp = tempfile.mkdtemp(prefix="csig_sub_")
        with zipfile.ZipFile(a.zip) as z:
            names = [n for n in z.namelist() if n.startswith("model_dir/")]
            if not names:
                sys.exit("zip 根目录下没有 model_dir/(层级错误会导致官方测评卡住)")
            z.extractall(tmp, members=names)
        a.model_dir = os.path.join(tmp, "model_dir"); print("unzipped model_dir ->", a.model_dir)
    label = a.label or os.path.basename(os.path.dirname(os.path.abspath(a.model_dir))) or "cand"
    out_dir = a.out_dir or f"render_{label}"
    res = {"label": label, "model_dir": os.path.abspath(a.model_dir)}
    import torch
    r, t_init = load_runner(a.model_dir)
    res["init_s"] = t_init; res["path"] = getattr(r, "_path", "?")
    print(f"[{label}] Runner init {t_init:.1f}s  path={res['path']}", flush=True)
    if not a.skip_latency:
        lat = latency_512(r); res["latency_512"] = lat
        print(f"[{label}] 512 fp16 单发: event med {lat['event_med']:.3f} p10 {lat['event_p10']:.3f} | wall med {lat['wall_med']:.3f} min {lat['wall_min']:.3f} ms", flush=True)
    if not a.skip_render:
        vd, td, t_r = render(r, out_dir, a.n, a.jpg_quality); res["render_s"] = t_r
        print(f"[{label}] rendered {a.n}+3 imgs in {t_r:.0f}s -> {out_dir}", flush=True)
    else:
        vd, td = os.path.join(out_dir, "val"), os.path.join(out_dir, "test")
    del r; torch.cuda.empty_cache()
    M = CM.create_metrics(torch.device("cuda"))
    v = score_val_dir(M, vd); print(f"[{label}] {fmt_val(v)}", flush=True)
    t = score_test_dir(M, td, a.n); print(f"[{label}] {fmt_test(t)}", flush=True)
    S = t["nr"] + v["fr"]; res.update({"val": v, "test": t, "S_proxy": S})
    line = f"[{label}] 代理 S = {t['nr']:.4f} + {v['fr']:.4f} = {S:.4f}"
    if "latency_512" in res:
        ms = res["latency_512"]["event_p10"]; g = ms * (ANCHOR["g"] / a.ref_ms) if a.ref_ms else ms
        p = predict(S, g); res["pred"] = p; res["g_used_ms"] = g
        line += f"   g={g:.3f}ms{'(已换算)' if a.ref_ms else '(未换算,按 AutoDL 口径)'}  预测榜分 ≈ {p:.1f}  (锚 G4 152.95)"
    print(line, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    jp = a.out or os.path.join(out_dir, "result.json")
    json.dump(res, open(jp, "w"), indent=2, ensure_ascii=False); print("saved", jp)


if __name__ == "__main__":
    main()
