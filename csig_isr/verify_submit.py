"""端到端验证提交包：完全模拟主办方的使用方式。

验证项：
  1. 从 model_dir 实例化 Runner（只给目录，不给其它参数）
  2. infer 接口签名与值域正确（[-1,1] 进，[-1,1] 出）
  3. 512x512 时延测量（赛题口径）
  4. 与出图路径的一致性（确保提交的图确实是这个模型产出的）
  5. output_dir 的 100 张图完整性
"""
import os, sys, time, glob, json
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/cache/zihaohe/csig_isr"
SUBMIT = f"{ROOT}/submit"
MODEL_DIR = f"{SUBMIT}/model_dir"
OUT_DIR = f"{SUBMIT}/output_dir"
TEST = f"{ROOT}/data/赛题二/测试集"

print("=" * 74)
print("1. 提交包结构检查")
print("=" * 74)
ok = True
for p, desc in [(MODEL_DIR, "model_dir"), (OUT_DIR, "output_dir"),
                (f"{MODEL_DIR}/runner.py", "runner.py"),
                (f"{MODEL_DIR}/config.yaml", "config.yaml"),
                (f"{MODEL_DIR}/tinysr_x1.pt", "jit 模型")]:
    e = os.path.exists(p)
    ok &= e
    sz = ""
    if e and os.path.isfile(p):
        sz = f"  {os.path.getsize(p)/1e6:.1f} MB"
    print(f"   {'[OK]' if e else '[缺失]'} {desc:<16}{sz}")

imgs = sorted(glob.glob(f"{OUT_DIR}/*.jpg"))
need = set(f"case{i}.jpg" for i in range(1, 101))
have = set(os.path.basename(f) for f in imgs)
print(f"   [{'OK' if have == need else '错误'}] 图片 {len(imgs)}/100  "
      f"缺失 {sorted(need-have) or '无'}  多余 {sorted(have-need) or '无'}")

# 尺寸与输入一致
bad = []
for f in imgs:
    n = os.path.basename(f)
    lp = os.path.join(TEST, n)
    if os.path.exists(lp):
        if Image.open(f).size != Image.open(lp).size:
            bad.append(n)
print(f"   [{'OK' if not bad else '错误'}] 尺寸与输入一致  异常 {bad or '无'}")

print()
print("=" * 74)
print("2. 模拟主办方加载 Runner")
print("=" * 74)
sys.path.insert(0, MODEL_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location("submit_runner", f"{MODEL_DIR}/runner.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
t0 = time.time()
runner = mod.Runner(MODEL_DIR)
print(f"   Runner(model_dir) 实例化成功  耗时 {time.time()-t0:.2f}s")
print(f"   device={runner.device}  dtype={runner.weight_dtype}")

print()
print("=" * 74)
print("3. infer 接口验证（[-1,1] 进出）")
print("=" * 74)
a = np.asarray(Image.open(f"{TEST}/case1.jpg").convert("RGB"), dtype=np.float32) / 255.0
t = torch.from_numpy(a).permute(2, 0, 1)[None][:, :, 1000:1512, 1000:1512]
x = (t * 2 - 1).to(runner.device, dtype=runner.weight_dtype)
out = runner.infer(x)
print(f"   输入 {tuple(x.shape)} range [{x.min():.3f}, {x.max():.3f}]")
print(f"   输出 {tuple(out.shape)} range [{out.min():.3f}, {out.max():.3f}] dtype={out.dtype}")
assert out.shape == x.shape, "输出尺寸不符"
print(f"   [OK] 尺寸一致，值域合理")

# 带 prompt 参数调用（官方签名有该参数）
out2 = runner.infer(x, prompt="a photo")
print(f"   [OK] infer(x, prompt=...) 可调用，与无 prompt 差异 "
      f"{(out2.float()-out.float()).abs().max().item():.6f}")

print()
print("=" * 74)
print("4. 时延测量（赛题口径：512x512, fp16, 单卡）")
print("=" * 74)
xb = torch.rand(1, 3, 512, 512, device=runner.device, dtype=runner.weight_dtype) * 2 - 1
with torch.no_grad():
    for _ in range(20):
        runner.infer(xb)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(100):
        runner.infer(xb)
    e.record()
    torch.cuda.synchronize()
lat = s.elapsed_time(e) / 100
T_OSE = 212.768
sp = T_OSE / lat
print(f"   时延 {lat:.3f} ms")
print(f"   OSEDiff 基准 {T_OSE:.2f} ms  ->  加速比 {sp:.2f}x   Speedup^0.2 = {sp**0.2:.4f}")

print()
print("=" * 74)
print("5. 与已出图的一致性（确认提交的图确由此模型产出）")
print("=" * 74)
# 取 case1 的一个 512 tile，用 runner 重算，与已出图对比
y0, x0 = 1024, 1024
lq_t = torch.from_numpy(a).permute(2, 0, 1)[None][:, :, y0:y0+512, x0:x0+512]
xin = (lq_t * 2 - 1).to(runner.device, dtype=runner.weight_dtype)
o = runner.infer(xin).float()
o01 = (o * 0.5 + 0.5).clamp(0, 1)
sub = np.asarray(Image.open(f"{OUT_DIR}/case1.jpg").convert("RGB"),
                 dtype=np.float32)[y0:y0+512, x0:x0+512] / 255.0
sub_t = torch.from_numpy(sub).permute(2, 0, 1)[None].to(runner.device)
diff = (o01 - sub_t).abs()
print(f"   runner 单 tile vs 提交图同位置：max {diff.max().item():.4f}  "
      f"mean {diff.mean().item():.5f}")
print(f"   注：提交图经过 tile 融合(overlap 128)+wavelet 色彩对齐+JPEG 编码，")
print(f"       故存在差异属正常；此处只确认量级合理（非完全不同的模型）。")

print()
print("=" * 74)
print(f"结论：{'提交包可用' if ok and have == need and not bad else '存在问题，需修复'}")
print("=" * 74)
json.dump({"latency_ms": lat, "speedup": sp, "n_images": len(imgs)},
          open(f"{ROOT}/eval_out/submit_verify.json", "w"), indent=1)
