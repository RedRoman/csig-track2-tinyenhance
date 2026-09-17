"""GT-aware high-frequency overshoot regularization and diagnostics.

The loss intentionally uses GT, not LQ, to decide whether a region is smooth.
An LQ region can be smooth simply because the degradation removed real detail;
using LQ as the mask would suppress exactly the content an enhancement model is
supposed to recover.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _odd(k: int) -> int:
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"kernel size must be positive odd, got {k}")
    return k


def _box_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    k = _odd(k)
    if k == 1:
        return x
    p = k // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), k, stride=1)


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    if x.shape[1] != 3:
        raise ValueError(f"expected 1 or 3 channels, got {x.shape[1]}")
    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def highpass_luma(x: torch.Tensor, blur_kernel: int = 5) -> torch.Tensor:
    y = rgb_to_luma(x.float())
    return y - _box_blur(y, blur_kernel)


def local_hf_energy(x: torch.Tensor, blur_kernel: int = 5,
                    local_kernel: int = 9) -> torch.Tensor:
    hp = highpass_luma(x, blur_kernel)
    return _box_blur(hp.square(), local_kernel)


def smooth_region_mask(gt_energy: torch.Tensor, quantile: float = 0.35,
                       softness: float = 0.20, absolute_cap: float = 0.0) -> torch.Tensor:
    """Soft mask for the smoothest GT regions in each sample.

    ``absolute_cap`` (if >0) prevents a globally textured image from declaring a
    high-energy region "smooth" merely because it lies in the bottom quantile.
    """
    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("smooth quantile must be in (0,1)")
    flat = gt_energy.detach().flatten(1)
    tau = torch.quantile(flat, q, dim=1, keepdim=True).view(-1, 1, 1, 1)
    if absolute_cap > 0:
        tau = torch.minimum(tau, tau.new_full(tau.shape, float(absolute_cap)))
    scale = (tau.abs() * float(softness)).clamp_min(1e-6)
    return torch.sigmoid((tau - gt_energy.detach()) / scale)


def gt_aware_hf_overshoot_loss(sr: torch.Tensor, gt: torch.Tensor, *,
                               blur_kernel: int = 5, local_kernel: int = 9,
                               margin: float = 0.35, slack: float = 2e-4,
                               smooth_quantile: float = 0.35,
                               mask_softness: float = 0.20,
                               smooth_abs_cap: float = 0.0):
    """Penalize only excess SR high-frequency energy in GT-smooth regions.

    Target budget: ``E_sr <= (1 + margin) * E_gt + slack``.
    Returns ``(loss, diagnostics_dict)``. No inference-time module is introduced.
    """
    sr_e = local_hf_energy(sr, blur_kernel, local_kernel)
    gt_e = local_hf_energy(gt.detach(), blur_kernel, local_kernel)
    mask = smooth_region_mask(gt_e, smooth_quantile, mask_softness, smooth_abs_cap)
    budget = (1.0 + float(margin)) * gt_e + float(slack)
    excess = F.relu(sr_e - budget)
    denom = mask.sum().clamp_min(1.0)
    loss = (excess * mask).sum() / denom
    with torch.no_grad():
        eps = 1e-8
        masked_sr = (sr_e * mask).sum() / denom
        masked_gt = (gt_e * mask).sum() / denom
        diag = {
            "hf_sr_gt_ratio": float((sr_e.mean() / (gt_e.mean() + eps)).item()),
            "hf_smooth_sr_gt_ratio": float((masked_sr / (masked_gt + eps)).item()),
            "hf_overshoot": float(loss.detach().item()),
            "hf_mask_mean": float(mask.mean().item()),
        }
    return loss, diag


@torch.no_grad()
def hf_diagnostic_maps(sr: torch.Tensor, gt: torch.Tensor, *,
                       blur_kernel: int = 5, local_kernel: int = 9,
                       margin: float = 0.35, slack: float = 2e-4,
                       smooth_quantile: float = 0.35,
                       mask_softness: float = 0.20,
                       smooth_abs_cap: float = 0.0):
    """Return detached energy/mask/excess maps for offline diagnosis."""
    sr_e = local_hf_energy(sr, blur_kernel, local_kernel)
    gt_e = local_hf_energy(gt, blur_kernel, local_kernel)
    mask = smooth_region_mask(gt_e, smooth_quantile, mask_softness, smooth_abs_cap)
    budget = (1.0 + float(margin)) * gt_e + float(slack)
    excess = F.relu(sr_e - budget)
    return {
        "sr_energy": sr_e.detach(),
        "gt_energy": gt_e.detach(),
        "smooth_mask": mask.detach(),
        "excess": excess.detach(),
    }
