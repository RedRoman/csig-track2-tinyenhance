"""把 Samsung SRNet 从 x4 (128->512) 改造为 x1 (512->512)。

原理：
  x4:  512 输入 -> PixelUnshuffle(2) -> 64x64x12  -> UNet -> 64x64x4 -> TAESD(8x) -> 512
       (输入 128, 下采 2x, 解码上采 8x, 净 4x)
  x1:  512 输入 -> PixelUnshuffle(8) -> 64x64x192 -> UNet -> 64x64x4 -> TAESD(8x) -> 512
       (输入 512, 下采 8x, 解码上采 8x, 净 1x)

latent 尺寸 64x64 完全不变 => UNet body / refine / decoder 权重 100% 复用，
只有 conv_in 需要从 12 通道扩到 192 通道。

conv_in 初始化采用空间对应映射而非朴素 repeat：
  PixelUnshuffle(r) 输出通道序为 c*r^2 + i*r + j
  new(c,i,j) 映射到 old(c, i//4, j//4)，权重除以 16
  这样当 512 图恰为 128 图的 8x 最近邻放大时，新旧网络输出完全一致。
"""
import os, sys, time
import torch
import torch.nn as nn

sys.path.insert(0, "/cache/zihaohe/csig_isr/NTIRE2026")
from diffusers import UNet2DConditionModel
from models.team11_SamsungAKCamera.model import create_srnet

CKPT = "/cache/zihaohe/csig_isr/NTIRE2026/model_zoo/team11_SamsungAICamera/team11_mobile_samsung/weights.pth"
SD21 = "/cache/zihaohe/csig_isr/sd21base"
OUT = "/cache/zihaohe/csig_isr/x1_model"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")


def build_x4():
    cfg = UNet2DConditionModel.load_config(SD21, subfolder="unet")
    unet = UNet2DConditionModel.from_config(cfg)
    return create_srnet(unet, checkpoint_path=CKPT)


def expand_conv_in(old_conv, r_old=2, r_new=8):
    """把 conv_in 从 PixelUnshuffle(r_old) 的通道数扩到 r_new 的通道数。"""
    C = 3
    cin_old, cout = old_conv.in_channels, old_conv.out_channels
    assert cin_old == C * r_old * r_old, f"意外的 conv_in 输入通道 {cin_old}"
    cin_new = C * r_new * r_new
    ratio = (r_new // r_old) ** 2          # 每个旧通道被拆成多少新通道

    new_conv = nn.Conv2d(cin_new, cout, old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding,
                         bias=old_conv.bias is not None)
    with torch.no_grad():
        W = torch.zeros(cout, cin_new, *old_conv.kernel_size,
                        dtype=old_conv.weight.dtype)
        s = r_new // r_old                 # 空间缩放比 = 4
        for c in range(C):
            for i in range(r_new):
                for j in range(r_new):
                    new_idx = c * r_new * r_new + i * r_new + j
                    old_idx = c * r_old * r_old + (i // s) * r_old + (j // s)
                    W[:, new_idx] = old_conv.weight[:, old_idx] / ratio
        new_conv.weight.copy_(W)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    return new_conv


print("=" * 78)
print("1. 载入原始 x4 模型")
print("=" * 78)
m = build_x4()
inner = m.net.body[1]
print(f"   PixelUnshuffle: {m.net.body[0]}")
print(f"   conv_in       : {inner.conv_in}")

print()
print("=" * 78)
print("2. 执行 x1 改造")
print("=" * 78)
new_conv_in = expand_conv_in(inner.conv_in, 2, 8)
inner.conv_in = new_conv_in
m.net.body[0] = nn.PixelUnshuffle(8)
print(f"   PixelUnshuffle -> {m.net.body[0]}")
print(f"   conv_in        -> {inner.conv_in}")

n_tot = sum(p.numel() for p in m.parameters())
print(f"   改造后总参数量 {n_tot/1e6:.2f} M")

m = m.to(dev).to(torch.float16).eval()

print()
print("=" * 78)
print("3. 尺寸流验证 (512 -> 512)")
print("=" * 78)
with torch.no_grad():
    x = torch.rand(1, 3, 512, 512, device=dev, dtype=torch.float16)
    z0 = m.net(x);       print(f"   net      {tuple(x.shape)} -> {tuple(z0.shape)}")
    z1 = m.refine(z0);   print(f"   refine   {tuple(z0.shape)} -> {tuple(z1.shape)}")
    sr = m.decoder(z1);  print(f"   decoder  {tuple(z1.shape)} -> {tuple(sr.shape)}")
    out = m(x)
    ok = out.shape == x.shape
    print(f"   完整     {tuple(x.shape)} -> {tuple(out.shape)}   {'尺寸正确' if ok else '尺寸错误'}")
    print(f"   输出范围 [{out.min().item():.4f}, {out.max().item():.4f}]")

print()
print("=" * 78)
print("4. 初始化一致性检查")
print("   (512 图 = 128 图的 4x 最近邻放大时, x1 与 x4 的 latent 应完全一致)")
print("=" * 78)
m4 = build_x4().to(dev).to(torch.float16).eval()
with torch.no_grad():
    small = torch.rand(1, 3, 128, 128, device=dev, dtype=torch.float16)
    big = torch.nn.functional.interpolate(small, scale_factor=4, mode="nearest")
    z_x4 = m4.net(small)
    z_x1 = m.net(big)
    print(f"   x4: 输入 {tuple(small.shape)} -> latent {tuple(z_x4.shape)}")
    print(f"   x1: 输入 {tuple(big.shape)} -> latent {tuple(z_x1.shape)}")
    d = (z_x4.float() - z_x1.float()).abs()
    print(f"   x4 latent 均值 {z_x4.float().mean().item():+.5f}  标准差 {z_x4.float().std().item():.5f}")
    print(f"   x1 latent 均值 {z_x1.float().mean().item():+.5f}  标准差 {z_x1.float().std().item():.5f}")
    print(f"   绝对差 max {d.max().item():.5f}  mean {d.mean().item():.5f}")
    rel = d.mean().item() / max(z_x4.float().abs().mean().item(), 1e-6)
    print(f"   相对误差 {rel*100:.3f}%   {'初始化正确' if rel < 0.05 else '初始化有偏差(fp16 舍入或映射错误)'}")

print()
print("=" * 78)
print("5. x1 模型时延 (512x512, 即赛题测速口径)")
print("=" * 78)


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


with torch.no_grad():
    x = torch.rand(1, 3, 512, 512, device=dev, dtype=torch.float16)
    z0 = m.net(x); z1 = m.refine(z0)

t_net = bench(m.net, x)
t_ref = bench(m.refine, z0)
t_dec = bench(m.decoder, z1)
t_all = bench(m, x)
print(f"   {'模块':<24}{'ms':>9}{'占比':>9}")
for nm, t in [("net (剪枝UNet)", t_net), ("refine", t_ref), ("decoder (TAESD)", t_dec)]:
    print(f"   {nm:<24}{t:>9.3f}{t/t_all*100:>8.1f}%")
print(f"   {'-'*42}")
print(f"   {'完整 forward':<24}{t_all:>9.3f}{100.0:>8.1f}%")

torch.save({"net": m.net.state_dict(), "refine": m.refine.state_dict(),
            "decoder": m.decoder.state_dict()},
           os.path.join(OUT, "srnet_x1_init.pth"))
print()
print(f"x1 初始化权重已保存到 {OUT}/srnet_x1_init.pth")
