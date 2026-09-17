"""定性 jit 数值差异：是 fp16 舍入 + freeze 算子融合，还是真的静默失败？

判据：若 fp32 下 trace 与 eager 差异 ~1e-7 (机器精度)，则 fp16 下的 1e-4 是舍入而非逻辑错误。
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
dev = torch.device("cuda")


def build_x1():
    cfg = UNet2DConditionModel.load_config(SD21, subfolder="unet")
    unet = UNet2DConditionModel.from_config(cfg)
    m = create_srnet(unet, checkpoint_path=CKPT)
    m.net.body[1].conv_in = nn.Conv2d(192, 160, 3, padding=1)
    m.net.body[0] = nn.PixelUnshuffle(8)
    sd = torch.load(X1, map_location="cpu", weights_only=False)
    m.net.load_state_dict(sd["net"]); m.refine.load_state_dict(sd["refine"])
    m.decoder.load_state_dict(sd["decoder"])
    return m


def report(tag, a, b):
    d = (a - b).abs()
    rel = d.mean().item() / max(a.abs().mean().item(), 1e-12)
    print(f"   {tag:<34} max {d.max().item():.3e}  mean {d.mean().item():.3e}  相对 {rel*100:.6f}%")
    return rel


print("=" * 78)
print("A. fp32 下 trace vs eager (排除 fp16 因素)")
print("=" * 78)
m32 = build_x1().to(dev).float().eval()
x32 = torch.rand(1, 3, 512, 512, device=dev, dtype=torch.float32)
with torch.no_grad():
    ref32 = m32(x32).clone()
    tr32 = torch.jit.trace(m32, x32, check_trace=False)
    tr32_frozen = torch.jit.freeze(tr32.eval())
    for _ in range(3):
        _ = tr32_frozen(x32)
    out_traced = tr32(x32).clone()
    out_frozen = tr32_frozen(x32).clone()

r1 = report("trace (未 freeze) vs eager", ref32, out_traced)
r2 = report("trace + freeze vs eager", ref32, out_frozen)

print()
print("=" * 78)
print("B. fp16 下逐模块定位差异来源")
print("=" * 78)
m16 = build_x1().to(dev).half().eval()
x16 = torch.rand(1, 3, 512, 512, device=dev, dtype=torch.float16)
with torch.no_grad():
    z0 = m16.net(x16); z1 = m16.refine(z0); ref16 = m16.decoder(z1).float()
    tr_net = torch.jit.freeze(torch.jit.trace(m16.net, x16, check_trace=False).eval())
    tr_ref = torch.jit.freeze(torch.jit.trace(m16.refine, z0, check_trace=False).eval())
    tr_dec = torch.jit.freeze(torch.jit.trace(m16.decoder, z1, check_trace=False).eval())
    for _ in range(3):
        _ = tr_net(x16); _ = tr_ref(z0); _ = tr_dec(z1)
    jz0 = tr_net(x16); jz1 = tr_ref(z0); jdec = tr_dec(z1).float()

report("net    (剪枝UNet)", z0.float(), jz0.float())
report("refine (8层卷积)", z1.float(), jz1.float())
report("decoder(TAESD)", ref16, jdec)

print()
print("=" * 78)
print("C. fp16 eager 自身的非确定性 (同输入重复跑)")
print("=" * 78)
with torch.no_grad():
    a = m16(x16).float().clone()
    b = m16(x16).float().clone()
report("eager 两次调用", a, b)

print()
print("=" * 78)
print("D. fp16 vs fp32 eager 的固有差异 (fp16 精度上限)")
print("=" * 78)
with torch.no_grad():
    o16 = m16(x16).float()
    o32 = m32(x16.float())
report("fp16 eager vs fp32 eager", o32, o16)

print()
print("=" * 78)
print("结论")
print("=" * 78)
if r2 < 1e-5:
    print("   fp32 下 trace+freeze 与 eager 一致 -> trace 逻辑正确, global skip 被正确内联")
    print("   fp16 下的 ~1e-4 差异来自 fp16 舍入与 freeze 算子融合, 属正常范围")
    print("   判定: 可安全用于提交")
else:
    print(f"   fp32 下仍有 {r2*100:.4f}% 差异 -> trace 存在逻辑问题, 需排查")
