"""预下载 pyiqa 各指标权重（GitHub 直连不稳，走 gh-proxy）。"""
import os, sys, subprocess, importlib

CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(CACHE, exist_ok=True)

# 需要的指标 -> 其 arch 模块
NEEDED = {
    "dists": "dists_arch",
    "niqe": "niqe_arch",
    "clipiqa": "clipiqa_arch",
    "musiq": "musiq_arch",
    "maniqa": "maniqa_arch",
    "topiq_nr": "topiq_arch",
}

urls = set()
for name, mod in NEEDED.items():
    try:
        m = importlib.import_module(f"pyiqa.archs.{mod}")
        d = getattr(m, "default_model_urls", None)
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str) and v.startswith("http"):
                    urls.add(v)
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, str) and vv.startswith("http"):
                            urls.add(vv)
    except Exception as e:
        print(f"  [{name}] 读取 url 失败: {str(e)[:60]}")

# CLIP / 其它隐式依赖
urls.add("https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CLIP-ViT-B-32.pt")

print(f"共 {len(urls)} 个 URL 待下载\n")
ok = fail = skip = 0
for u in sorted(urls):
    fn = os.path.basename(u.split("?")[0])
    dst = os.path.join(CACHE, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        print(f"  [跳过] {fn} ({os.path.getsize(dst)/1e6:.1f} MB)")
        skip += 1
        continue
    got = False
    for prefix in ["https://gh-proxy.com/", ""]:
        cmd = ["curl", "-sL", "--max-time", "300", "-o", dst, prefix + u]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 10000:
            print(f"  [成功] {fn} ({os.path.getsize(dst)/1e6:.1f} MB) via {'gh-proxy' if prefix else '直连'}")
            ok += 1
            got = True
            break
    if not got:
        print(f"  [失败] {fn}\n         {u}")
        fail += 1

print(f"\n成功 {ok} / 跳过 {skip} / 失败 {fail}")
print(f"缓存目录: {CACHE}")
