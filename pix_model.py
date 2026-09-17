"""像素空间一步式去噪 UNet(moonshot 学生)。pixel-unshuffle×ps 后在低分辨率工作,时间步嵌入 AdaGN 调制。
接口与 TinySRCore 对齐:forward(lq01)->out01, forward_train(lq01)->(out01, alpha);core(x_m11)->y_m11 供 runner。"""
import math
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

    def core_feat(s, x, t=None):
        """同 core,另返回瓶颈(mid 末)特征 [B, chs[-1], H/(ps*4), W/(ps*4)],供特征蒸馏。"""
        if t is None: t = torch.full((x.shape[0],), s.t_fixed, device=x.device, dtype=x.dtype)
        half = s.tdim // 2
        fr = torch.exp(-math.log(1e4) * torch.arange(half, device=x.device, dtype=x.dtype) / half)
        te = s.temb(torch.cat([torch.sin(t[:, None] * fr), torch.cos(t[:, None] * fr)], 1))
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


def build_pix(ckpt=None, device="cuda", dtype=torch.float16, **kw):
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        m = PixUNet(**sd["cfg"]); m.load_state_dict(sd["state"], strict=True)
    else:
        m = PixUNet(**kw)
    return m.to(device, dtype).eval()
