# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted channel-only LayerNorm: equivalent affine normalization over C.
# FP16/BF16 accumulate statistics in FP32; float64 is preserved for CPU checks.
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        z = x.to(dtype)
        mean = z.mean(dim=1, keepdim=True)
        variance = (z - mean).square().mean(dim=1, keepdim=True)
        y = (z - mean) * torch.rsqrt(variance + self.eps)
        y = y * self.weight.to(dtype)[None,:,None,None] + self.bias.to(dtype)[None,:,None,None]
        return y.to(x.dtype)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


ARCHITECTURE = dict(img_channel=3, width=32, middle_blk_num=1,
                    enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])
ORIGINAL_SOURCE_SHA256 = '01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967'
ORIGINAL_SOURCE_URL = 'https://raw.githubusercontent.com/megvii-research/NAFNet/2b4af71ebe098a92a75910c233a3965a3e93ede4/basicsr/models/archs/NAFNet_arch.py'


def build_model(zero_ending=True):
    model = NAFNet(**ARCHITECTURE)
    if zero_ending:
        nn.init.zeros_(model.ending.weight)
        nn.init.zeros_(model.ending.bias)
    return model


def tile_positions(length, core=512, stride=256):
    end = max(length-core, 0)
    values = list(range(0, end+1, stride))
    if values[-1] != end: values.append(end)
    return values


def native_tiled(model, x, gaussian, core=512, halo=64, stride=256):
    # Every input follows fixed640 contexts, including a512 benchmark input.
    n,c,h,w = x.shape
    if (n,c) != (1,3): raise ValueError('native tiler requires batch1 RGB')
    ph,pw = max(core-h,0),max(core-w,0)
    # Replicate is defined for arbitrary tiny inputs; canonical images exceed512.
    base = F.pad(x,(0,pw,0,ph),mode='replicate') if ph or pw else x
    hp,wp = base.shape[-2:]
    padded = F.pad(base,(halo,halo,halo,halo),mode='reflect')
    model_dtype = next(model.parameters()).dtype
    accum = torch.zeros((1,3,hp,wp),device=x.device,dtype=torch.float32)
    weights = torch.zeros((1,1,hp,wp),device=x.device,dtype=torch.float32)
    for top in tile_positions(hp,core,stride):
        for left in tile_positions(wp,core,stride):
            context = padded[...,top:top+core+2*halo,left:left+core+2*halo].to(model_dtype)
            y = model(context)
            # Average learned residuals, retaining the native FP32 identity path.
            # With a zero ending this is exactly x, including FP16 NN inference.
            delta = (y[...,halo:halo+core,halo:halo+core].float()
                     - context[...,halo:halo+core,halo:halo+core].float())
            accum[...,top:top+core,left:left+core] += delta * gaussian
            weights[...,top:top+core,left:left+core] += gaussian
    return (base.float() + accum / weights.clamp_min(1e-12))[...,:h,:w]
