"""Samsung SRNet 冒烟测试：加载权重 → 前向 → 逐模块 profiling → 尺寸流验证。"""
import os, sys, time
import torch

sys.path.insert(0, "/cache/zihaohe/csig_isr/NTIRE2026")

from diffusers import UNet2DConditionModel
from models.team11_SamsungAKCamera.model import create_srnet

CKPT = "/cache/zihaohe/csig_isr/NTIRE2026/model_zoo/team11_SamsungAICamera/team11_mobile_samsung/weights.pth"
SD21 = "/cache/zihaohe/csig_isr/sd21base"
dev = torch.device("cuda")

print("=" * 78)
print("1. 构造 UNet 结构（仅用 SD2.1 config，不加载 SD 权重）")
print("=" * 78)
config = UNet2DConditionModel.load_config(SD21, subfolder="unet")
unet = UNet2DConditionModel.from_config(config)
print(f"   UNet from_config OK, 原始参数量 {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")

print()
print("=" * 78)
print("2. 构造 SRNet 并加载 Samsung 权重")
print("=" * 78)
model = create_srnet(unet, checkpoint_path=CKPT).to(dev).eval()

n_net = sum(p.numel() for p in model.net.parameters())
n_ref = sum(p.numel() for p in model.refine.parameters())
n_dec = sum(p.numel() for p in model.decoder.parameters())
print(f"   net     (剪枝UNet)      {n_net/1e6:>8.2f} M")
print(f"   refine  (EightLayerConv){n_ref/1e6:>8.2f} M")
print(f"   decoder (TAESD)         {n_dec/1e6:>8.2f} M")
print(f"   ---------------------------------")
print(f"   总计                    {(n_net+n_ref+n_dec)/1e6:>8.2f} M")

model = model.to(dtype=torch.float16)

print()
print("=" * 78)
print("3. 尺寸流验证（原生 x4：128 -> 512）")
print("=" * 78)
with torch.no_grad():
    x = torch.rand(1, 3, 128, 128, device=dev, dtype=torch.float16)
    z0 = model.net(x);            print(f"   net(x)       {tuple(x.shape)} -> {tuple(z0.shape)}")
    z1 = model.refine(z0);        print(f"   refine(z0)   {tuple(z0.shape)} -> {tuple(z1.shape)}")
    sr = model.decoder(z1);       print(f"   decoder(z1)  {tuple(z1.shape)} -> {tuple(sr.shape)}")
    out = model(x);               print(f"   完整 forward {tuple(x.shape)} -> {tuple(out.shape)}")
    print(f"   输出范围 [{out.min().item():.3f}, {out.max().item():.3f}]")

print()
print("=" * 78)
print("4. 逐模块 profiling（512x512 输出口径，即赛题测速口径）")
print("=" * 78)


def bench(fn, inp, n=100, warm=20):
    with torch.no_grad():
        for _ in range(warm):
            fn(inp)
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn(inp)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / n


with torch.no_grad():
    x = torch.rand(1, 3, 128, 128, device=dev, dtype=torch.float16)
    z0 = model.net(x)
    z1 = model.refine(z0)

t_net = bench(model.net, x)
t_ref = bench(model.refine, z0)
t_dec = bench(model.decoder, z1)
t_all = bench(model, x)

print(f"   {'模块':<26}{'ms':>9}{'占比':>9}")
for name, t in [("net (剪枝UNet)", t_net), ("refine (8层卷积)", t_ref),
                ("decoder (TAESD)", t_dec)]:
    print(f"   {name:<26}{t:>9.3f}{t/t_all*100:>8.1f}%")
print(f"   {'-'*44}")
print(f"   {'完整 forward':<26}{t_all:>9.3f}{100.0:>8.1f}%")
print(f"   {'(三者之和)':<26}{t_net+t_ref+t_dec:>9.3f}")

print()
print("=" * 78)
print("5. x1 改造可行性预检（512 输入需要 PixelUnshuffle(8)）")
print("=" * 78)
body = model.net.body
print(f"   当前 net.body[0] = {body[0]}")
inner = body[1]
print(f"   当前 conv_in     = {inner.conv_in}")
print(f"   当前 conv_out    = {inner.conv_out}")
ps = torch.nn.PixelUnshuffle(8)
probe = ps(torch.rand(1, 3, 512, 512, device=dev, dtype=torch.float16))
print(f"   PixelUnshuffle(8) 对 512x512 -> {tuple(probe.shape)}")
print(f"   需要把 conv_in 从 {inner.conv_in.in_channels} 通道改为 {probe.shape[1]} 通道")
print(f"   latent 尺寸 {probe.shape[2]}x{probe.shape[3]} 与原生 x4 的一致 -> UNet/decoder 权重可全部复用")

print()
print("冒烟测试完成")
