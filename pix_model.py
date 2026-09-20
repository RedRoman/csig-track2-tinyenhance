"""像素空间一步式去噪 UNet(moonshot 学生)。pixel-unshuffle×ps 后在低分辨率工作,时间步嵌入 AdaGN 调制。
接口与 TinySRCore 对齐:forward(lq01)->out01, forward_train(lq01)->(out01, alpha);core(x_m11)->y_m11 供 runner。
v2(2026-09-18):PixUNet2 支持每层块数 nbs 与 norm∈{gn,none};fold_inference() 把固定时间步的调制常量折进权重,推理期零 temb 开销。"""
import math, copy
import torch, torch.nn as nn, torch.nn.functional as F


class Blk(nn.Module):
    def __init__(s, c, tdim):
        super().__init__()
        s.c1 = nn.Conv2d(c, c, 3, padding=1); s.c2 = nn.Conv2d(c, c, 3, padding=1)
        s.n = nn.GroupNorm(8, c); s.ada = nn.Linear(tdim, 2 * c)
        nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)
    def forward(s, x, te):
        sc, sh = s.ada(te)[:, :, None, None].chunk(2, 1)
        return x + s.c2(F.gelu(s.c1(F.gelu(s.n(x) * (1 + sc) + sh))))


class Blk2(nn.Module):
    """无归一化残差块:x + c2(gelu(c1(x) + b(te)))。推理期 b(te) 折进 c1.bias,块内仅 2 conv + 2 pointwise kernel。
    c2 零初始化(ReZero):起点恒等,无 norm 也稳。"""
    def __init__(s, c, tdim):
        super().__init__()
        s.c1 = nn.Conv2d(c, c, 3, padding=1); s.c2 = nn.Conv2d(c, c, 3, padding=1)
        s.ada = nn.Linear(tdim, c)
        nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)
        nn.init.zeros_(s.c2.weight); nn.init.zeros_(s.c2.bias)
    def forward(s, x, te):
        return x + s.c2(F.gelu(s.c1(x) + s.ada(te)[:, :, None, None]))


class PixUNet(nn.Module):
    def __init__(s, chs=(128, 256, 512), nb=3, tdim=128, ps=4, t_fixed=999.0):
        super().__init__()
        s.cfg = dict(chs=tuple(chs), nb=nb, tdim=tdim, ps=ps, t_fixed=t_fixed)
        s.tdim, s.ps, s.t_fixed = tdim, ps, float(t_fixed); cin = 3 * ps * ps
        s.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        s.inc = nn.Conv2d(cin, chs[0], 3, padding=1)
        s.enc = nn.ModuleList(); s.down = nn.ModuleList()
        for i, c in enumerate(chs):
            s.enc.append(nn.ModuleList([Blk(c, tdim) for _ in range(nb)]))
            if i < len(chs) - 1: s.down.append(nn.Conv2d(c, chs[i + 1], 3, stride=2, padding=1))
        s.mid = nn.ModuleList([Blk(chs[-1], tdim) for _ in range(nb)])
        s.up = nn.ModuleList(); s.dec = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            s.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 4, stride=2, padding=1))
            s.dec.append(nn.ModuleList([Blk(chs[i - 1], tdim) for _ in range(nb)]))
        s.outc = nn.Conv2d(chs[0], cin, 3, padding=1)
        nn.init.zeros_(s.outc.weight); nn.init.zeros_(s.outc.bias)   # 起点=恒等

    def core(s, x, t=None):
        """x: [-1,1]. 返回 [-1,1](未 clamp)。"""
        return s.core_feat(x, t)[0]

    def time_embed(s, t, device, dtype):
        half = s.tdim // 2
        fr = torch.exp(-math.log(1e4) * torch.arange(half, device=device, dtype=dtype) / half)
        return s.temb(torch.cat([torch.sin(t[:, None] * fr), torch.cos(t[:, None] * fr)], 1))

    def core_feat(s, x, t=None):
        """同 core,另返回瓶颈(mid 末)特征 [B, chs[-1], H/(ps*4), W/(ps*4)],供特征蒸馏。"""
        if t is None: t = torch.full((x.shape[0],), s.t_fixed, device=x.device, dtype=x.dtype)
        te = s.time_embed(t, x.device, x.dtype)
        x0 = x
        _, _, H, W = x.shape; p = s.ps * 4
        ph, pw = (-H) % p, (-W) % p
        if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        h = F.pixel_unshuffle(x, s.ps) if s.ps > 1 else x
        h = s.inc(h); skips = []
        for i, blks in enumerate(s.enc):
            for b in blks: h = b(h, te)
            skips.append(h)
            if i < len(s.down): h = s.down[i](h)
        for b in s.mid: h = b(h, te)
        mid = h
        for j, (up, blks) in enumerate(zip(s.up, s.dec)):
            h = up(h) + skips[-2 - j]
            for b in blks: h = b(h, te)
        r = s.outc(h)
        if s.ps > 1: r = F.pixel_shuffle(r, s.ps)
        if ph or pw: r = r[:, :, :H, :W]
        return x0 + r, mid

    def forward_train(s, lq01, return_feat=False):
        y, mid = s.core_feat(lq01 * 2 - 1)
        out = (y * 0.5 + 0.5).clamp(0, 1)
        if return_feat: return out, torch.ones(1, device=lq01.device), mid
        return out, torch.ones(1, device=lq01.device)

    def forward(s, lq01):
        return s.forward_train(lq01)[0]


class PixUNet2(PixUNet):
    """v2:每层块数 nbs(逐层,mid 取 nbs[-1],dec 镜像 enc),norm∈{gn(AdaGN 块),none(Blk2)}。其余与 PixUNet 同。"""
    def __init__(s, chs=(32, 64, 128), nbs=(1, 2, 2), norm="none", tdim=128, ps=2, t_fixed=999.0, nbs_dec=None):
        nn.Module.__init__(s)
        chs, nbs = tuple(int(c) for c in chs), tuple(int(n) for n in nbs)
        assert len(chs) == len(nbs) and norm in ("gn", "none")
        # nbs_dec: 解码侧每层块数(按 level 0..L-2 给出,与 enc 同序);None=镜像 enc。用于非对称剪解码器省时延。
        nbs_dec = tuple(int(n) for n in nbs_dec) if nbs_dec is not None else nbs[:-1]
        assert len(nbs_dec) == len(chs) - 1
        s.cfg = dict(chs=chs, nbs=nbs, norm=norm, tdim=tdim, ps=ps, t_fixed=t_fixed, arch_ver=2, nbs_dec=nbs_dec)
        s.tdim, s.ps, s.t_fixed, s.norm = tdim, ps, float(t_fixed), norm; cin = 3 * ps * ps
        B = (lambda c: Blk(c, tdim)) if norm == "gn" else (lambda c: Blk2(c, tdim))
        s.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        s.inc = nn.Conv2d(cin, chs[0], 3, padding=1)
        s.enc = nn.ModuleList(); s.down = nn.ModuleList()
        for i, c in enumerate(chs):
            s.enc.append(nn.ModuleList([B(c) for _ in range(nbs[i])]))
            if i < len(chs) - 1: s.down.append(nn.Conv2d(c, chs[i + 1], 3, stride=2, padding=1))
        s.mid = nn.ModuleList([B(chs[-1]) for _ in range(nbs[-1])])
        s.up = nn.ModuleList(); s.dec = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            s.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 4, stride=2, padding=1))
            s.dec.append(nn.ModuleList([B(chs[i - 1]) for _ in range(nbs_dec[i - 1])]))
        s.outc = nn.Conv2d(chs[0], cin, 3, padding=1)
        nn.init.zeros_(s.outc.weight); nn.init.zeros_(s.outc.bias)


def build_pix(ckpt=None, device="cuda", dtype=torch.float16, **kw):
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = dict(sd["cfg"]); ver = cfg.pop("arch_ver", 1)
        m = PixUNet2(**cfg) if ver == 2 else PixUNet(**cfg)
        m.load_state_dict(sd["state"], strict=True)
    else:
        m = PixUNet(**kw)
    return m.to(device, dtype).eval()


# ---------------- 推理期折叠(无损):固定 t → 调制量为常数 → 折进 GN affine / conv bias,删掉 temb 路径 ----------------
class _FoldBlkGN(nn.Module):
    def __init__(s, b, te):
        super().__init__()
        c = b.c1.in_channels; s.n = nn.GroupNorm(8, c)
        with torch.no_grad():
            sc, sh = b.ada(te)[0].chunk(2, 0)
            s.n.weight.copy_(b.n.weight * (1 + sc)); s.n.bias.copy_(b.n.bias * (1 + sc) + sh)
        s.c1, s.c2 = b.c1, b.c2
    def forward(s, x): return x + s.c2(F.gelu(s.c1(F.gelu(s.n(x)))))


class _FoldBlkNone(nn.Module):
    def __init__(s, b, te):
        super().__init__()
        s.c1, s.c2 = b.c1, b.c2
        with torch.no_grad(): s.c1.bias.add_(b.ada(te)[0])
    def forward(s, x): return x + s.c2(F.gelu(s.c1(x)))


class FoldedPix(nn.Module):
    """forward(x∈[-1,1]) -> clamp(-1,1)。与原模型 core(x).clamp(-1,1) 等价(fp16 舍入内)。"""
    def __init__(s, m):
        super().__init__()
        mf = copy.deepcopy(m).float().eval()
        dev = next(mf.parameters()).device
        with torch.no_grad():
            te = mf.time_embed(torch.full((1,), mf.t_fixed, device=dev, dtype=torch.float32), dev, torch.float32)
        FB = lambda b: _FoldBlkGN(b, te) if isinstance(b, Blk) else _FoldBlkNone(b, te)
        s.ps = mf.ps; s.inc = mf.inc; s.down = mf.down; s.up = mf.up; s.outc = mf.outc
        s.enc = nn.ModuleList([nn.Sequential(*[FB(b) for b in blks]) for blks in mf.enc])
        s.mid = nn.Sequential(*[FB(b) for b in mf.mid])
        s.dec = nn.ModuleList([nn.Sequential(*[FB(b) for b in blks]) for blks in mf.dec])
    def forward(s, x):
        x0 = x
        _, _, H, W = x.shape; p = s.ps * 4
        ph, pw = (-H) % p, (-W) % p
        if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        h = F.pixel_unshuffle(x, s.ps) if s.ps > 1 else x
        h = s.inc(h); skips = []
        for i, blks in enumerate(s.enc):
            h = blks(h); skips.append(h)
            if i < len(s.down): h = s.down[i](h)
        h = s.mid(h)
        for j, (up, blks) in enumerate(zip(s.up, s.dec)):
            h = blks(up(h) + skips[-2 - j])
        r = s.outc(h)
        if s.ps > 1: r = F.pixel_shuffle(r, s.ps)
        if ph or pw: r = r[:, :, :H, :W]
        return (x0 + r).clamp(-1, 1)


def fold_inference(m):
    """返回折叠后的推理模块(fp32,调用方自行 .to(device,dtype).eval())。"""
    return FoldedPix(m)
