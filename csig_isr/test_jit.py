"""torch.jit 导出验证（高风险项，赛题硬性要求）。

重点排查静默失败：
  1. model.py 用了模块级 `global skip` 传递 skip connection —— trace 能否正确内联
  2. fp16 + trace 的数值一致性
  3. 导出后独立 load 能否复现（模拟主办方环境）
  4. trace 后时延是否退化
"""
import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, "/cache/zihaohe/csig_isr/NTIRE2026")
from diffusers import UNet2DConditionModel
from models.team11_SamsungAKCamera.model import create_srnet

CKPT = "/cache/zihaohe/csig_isr/NTIRE2026/model_zoo/team11_SamsungAICamera/team11_mobile_samsung/weights.pth"
X1 = "/cache/zihaohe/csig_isr/x1_model/srnet_x1_init.pth"
SD21 = "/cache/zihaohe/csig_isr/sd21base"
OUT = "/cache/zihaohe/csig_isr/x1_model"
dev = torch.device("cuda")


def build_x1():
    cfg = UNet2DConditionModel.load_config(SD21, subfolder="unet")
    unet = UNet2DConditionModel.from_config(cfg)
    m = create_srnet(unet, checkpoint_path=CKPT)
    inner = m.net.body[1]
    inner.conv_in = nn.Conv2d(192, 160, 3, padding=1)
    m.net.body[0] = nn.PixelUnshuffle(8)
    sd = torch.load(X1, map_location="cpu", weights_only=False)
    m.net.load_state_dict(sd["net"])
    m.refine.load_state_dict(sd["refine"])
    m.decoder.load_state_dict(sd["decoder"])
    return m


def bench(fn, inp, n=100, warm=20):
    with torch.no_grad():
        for _ in range(warm):
            fn(inp)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn(inp)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / n


print("=" * 78)
print("1. 载入 x1 模型 (fp16, eval)")
print("=" * 78)
m = build_x1().to(dev).to(torch.float16).eval()
print(f"   参数量 {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

x = torch.rand(1, 3, 512, 512, device=dev, dtype=torch.float16)
with torch.no_grad():
    ref = m(x).float().clone()
print(f"   eager 输出 {tuple(ref.shape)}  范围 [{ref.min():.4f}, {ref.max():.4f}]")

print()
print("=" * 78)
print("2. torch.jit.trace 导出")
print("   注意: 绝不用 autocast; 输入 dtype 必须与权重一致")
print("=" * 78)
with torch.no_grad():
    traced = torch.jit.trace(m, x, check_trace=False)
    traced = torch.jit.freeze(traced.eval())
print("   trace + freeze 完成")

path = os.path.join(OUT, "srnet_x1.pt")
traced.save(path)
print(f"   已保存 {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

print()
print("=" * 78)
print("3. 独立加载复现 (模拟主办方环境)")
print("=" * 78)
del traced
torch.cuda.empty_cache()
loaded = torch.jit.load(path, map_location=dev)
loaded.eval()
with torch.no_grad():
    for _ in range(3):
        out = loaded(x)
    out = loaded(x).float()
d = (out - ref).abs()
rel = d.mean().item() / max(ref.abs().mean().item(), 1e-8)
print(f"   jit 输出 {tuple(out.shape)}  范围 [{out.min():.4f}, {out.max():.4f}]")
print(f"   与 eager 差异: max {d.max().item():.6f}  mean {d.mean().item():.8f}  相对 {rel*100:.5f}%")
print(f"   {'数值一致' if rel < 1e-4 else '数值不一致 - 存在静默失败!'}")

print()
print("=" * 78)
print("4. 多次调用一致性 (排查 global skip 状态污染)")
print("=" * 78)
with torch.no_grad():
    o1 = loaded(x).float().clone()
    xb = torch.rand(1, 3, 512, 512, device=dev, dtype=torch.float16)
    _ = loaded(xb)
    o2 = loaded(x).float().clone()
dd = (o1 - o2).abs().max().item()
print(f"   同输入两次调用 (中间插入不同输入) 最大差异 {dd:.8f}")
print(f"   {'无状态污染' if dd < 1e-5 else 'global skip 导致状态污染!'}")

print()
print("=" * 78)
print("5. 时延对比 (512x512 fp16)")
print("=" * 78)
t_eager = bench(m, x)
t_jit = bench(loaded, x)
print(f"   eager      {t_eager:>8.3f} ms")
print(f"   jit frozen {t_jit:>8.3f} ms   ({(t_eager/t_jit-1)*100:+.1f}% vs eager)")

print()
print("=" * 78)
print("6. 不同分辨率鲁棒性 (4K 分块时可能出现非 512 边块)")
print("=" * 78)
for size in [256, 512, 768]:
    xt = torch.rand(1, 3, size, size, device=dev, dtype=torch.float16)
    try:
        with torch.no_grad():
            o = loaded(xt)
        print(f"   {size}x{size} -> {tuple(o.shape)}  OK")
    except Exception as ex:
        print(f"   {size}x{size} -> 失败: {str(ex)[:60]}")

print()
print("jit 导出验证完成")
