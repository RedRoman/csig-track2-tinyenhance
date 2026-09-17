"""小学生：TAESD latent 空间里的一步扩散残差网络。

设计依据（全部来自实测）：
  - TinySR 整网 12.75ms = TAESD enc 2.28 + DiT(340M) 6.96 + TAESD dec 3.68
  - enc/dec 共 5.96ms 是地板（保持 TAESD 不动，它的 latent 空间就是教师的空间）
  - 预算：DiT 换成 <=1ms 的小网络 → 整网 ~7ms；若后续再蒸 TAESD dec 可到 ~5ms
  - 分数账：榜分 = 3.357 × 2^Score × spd^0.2
      Score 4.05 + 7ms(14.3x) → 3.357×16.56×1.70 = 94.5
      Score 4.05 + 5ms(20x)   → 101.7   ← 100 分线
  - 架构约束：赛题要求 Diffusion 架构。保持「VAE encode → 一步去噪(z - alpha*pred)
    → VAE decode」的一步扩散形式，只是 pred 网络从 340M DiT 换成小卷积网。
    与 TinySR 的接口完全同构，蒸馏配方（distill_v1 已验证 Score 轴保留率 100%）直接复用。

结构：latent 域 UNet 风格残差网，无注意力（64x64 的 latent 上卷积感受野足够，
注意力在这个尺寸下性价比极低），GroupNorm + SiLU，深度可分离用在中段省算力。
参数量目标 3-8M，512 输入下前向 <=1ms。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class TinyLatentNet(nn.Module):
    """z(16,64,64) -> pred(16,64,64)，一步扩散的速度/残差预测器。

    三个尺度：64 -> 32 -> 16，通道 C -> 2C -> 4C。
    最后一层零初始化：起点为恒等（pred=0 => zo=z），训练从教师那里学扰动。
    """

    def __init__(self, in_ch=16, base=64, blocks=(2, 2, 3)):
        super().__init__()
        C = base
        self.inp = nn.Conv2d(in_ch, C, 3, padding=1)
        self.d1 = nn.Sequential(*[ConvBlock(C) for _ in range(blocks[0])])
        self.down1 = nn.Conv2d(C, 2 * C, 3, stride=2, padding=1)
        self.d2 = nn.Sequential(*[ConvBlock(2 * C) for _ in range(blocks[1])])
        self.down2 = nn.Conv2d(2 * C, 4 * C, 3, stride=2, padding=1)
        self.mid = nn.Sequential(*[ConvBlock(4 * C) for _ in range(blocks[2])])
        self.up2 = nn.ConvTranspose2d(4 * C, 2 * C, 4, stride=2, padding=1)
        self.u2 = nn.Sequential(*[ConvBlock(2 * C) for _ in range(blocks[1])])
        self.up1 = nn.ConvTranspose2d(2 * C, C, 4, stride=2, padding=1)
        self.u1 = nn.Sequential(*[ConvBlock(C) for _ in range(blocks[0])])
        self.out = nn.Conv2d(C, in_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        h1 = self.d1(self.inp(z))
        h2 = self.d2(self.down1(h1))
        m = self.mid(self.down2(h2))
        x2 = self.u2(self.up2(m) + h2)
        x1 = self.u1(self.up1(x2) + h1)
        return self.out(x1)


class TinyStudent(nn.Module):
    """完整学生：TAESD encode -> z - alpha*TinyLatentNet(z) -> TAESD decode。
    与 TinySRCore 的 _impl 同构，alpha 固定 1（网络自己学幅度，零初始化保证起点恒等）。
    """

    def __init__(self, vae, base=64, blocks=(2, 2, 3)):
        super().__init__()
        self.vae = vae            # 冻结的 TAESD（separable，与教师/现役完全一致）
        self.net = TinyLatentNet(16, base, blocks)

    def forward(self, lq01):
        x = lq01 * 2 - 1
        sf = self.vae.config.scaling_factor
        z = self.vae.encode(x).latents * sf
        net_dtype = next(self.net.parameters()).dtype
        pred = self.net(z.to(net_dtype)).to(z.dtype)
        zo = z - pred
        img = self.vae.decode(zo / sf, return_dict=False)[0]
        return (img * 0.5 + 0.5).clamp(0, 1)


def build_student(base=64, blocks=(2, 2, 3), device="cuda", dtype=torch.float16):
    import os, sys
    sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
    from settings import TS
    if TS not in sys.path:
        sys.path.insert(0, TS)
    from models.vae.autoencoder_tiny import AutoencoderTiny
    vae = AutoencoderTiny.from_pretrained(
        os.path.join(TS, "checkpoint", "vae", "separable"), torch_dtype=dtype)
    vae = vae.to(device, dtype=dtype).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    m = TinyStudent(vae, base, blocks).to(device)
    m.net = m.net.to(dtype=torch.float32)   # 小网络 fp32 训练，推理时再转 fp16
    return m


if __name__ == "__main__":
    import time, numpy as np
    DEV = torch.device("cuda")
    for base, blocks in [(48, (2, 2, 2)), (64, (2, 2, 3)), (96, (2, 3, 4))]:
        m = build_student(base, blocks, DEV)
        n = sum(p.numel() for p in m.net.parameters()) / 1e6
        # 时延：fp16 + 512 输入
        net16 = TinyLatentNet(16, base, blocks).to(DEV, dtype=torch.float16).eval()
        z = torch.randn(1, 16, 64, 64, device=DEV, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(20):
                net16(z)
            torch.cuda.synchronize()
            ts = []
            for _ in range(100):
                s = torch.cuda.Event(True); e = torch.cuda.Event(True)
                s.record(); net16(z); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
        t_net = float(np.median(ts))
        total = 2.28 + t_net + 3.68
        spd = 100.15 / total
        print(f"base={base} blocks={blocks}: {n:.2f}M 参数, net {t_net:.2f}ms, "
              f"整网约 {total:.2f}ms (加速比 {spd:.1f}x, spd^0.2={spd**0.2:.3f})", flush=True)
        # 恒等起点验证
        x = torch.rand(1, 3, 512, 512, device=DEV, dtype=torch.float16)
        with torch.no_grad():
            y = m(x)
        sf = m.vae.config.scaling_factor
        with torch.no_grad():
            z0 = m.vae.encode(x * 2 - 1).latents * sf
            rt = (m.vae.decode(z0 / sf, return_dict=False)[0] * 0.5 + 0.5).clamp(0, 1)
        diff = float((y - rt).abs().max())
        print(f"  恒等起点检查：|student - vae往返| max = {diff:.2e}（应为 0）", flush=True)
        del m, net16
        torch.cuda.empty_cache()
