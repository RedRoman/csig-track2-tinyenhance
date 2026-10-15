"""Formal FP32 Runner/PNG roundtrip, with an identity straight-through gradient.

Use eager FP32 operations: compiling/fusing this function may alter thresholds.
Inputs must be finite. Build the LUT once and reuse it on the training device.
"""
import numpy as np
import torch


def make_png_lut(device=None):
    """Return the exact 256 FP32 values produced by CM.load_img's uint8/255."""
    values = np.arange(256, dtype=np.float32) / np.float32(255)
    return torch.from_numpy(values).to(device=device)


def png_quantization_ste(rgb, lut=None):
    """Forward: formal PNG decoded RGB. Backward: identity with respect to rgb."""
    if rgb.dtype != torch.float32:
        raise TypeError('Formal PNG surrogate requires FP32 rgb')
    if lut is None:
        lut = make_png_lut(rgb.device)
    if lut.dtype != torch.float32 or lut.device != rgb.device or tuple(lut.shape) != (256,):
        raise ValueError('LUT must have shape(256), FP32 dtype and match rgb.device')
    with torch.no_grad():
        # Keep these separate, matching Runner then NumPy renderer operations.
        normalized = rgb * 2.0
        normalized = normalized - 1.0
        normalized = normalized.clamp(-1.0, 1.0)
        encoded = normalized * 0.5
        encoded = encoded + 0.5
        encoded = encoded * 255.0
        indices = torch.round(encoded).to(torch.long)
        q = lut[indices]
    # rgb-rgb.detach() is exactly zero forward for finite rgb; gradient is one.
    return q.detach() + (rgb - rgb.detach())
