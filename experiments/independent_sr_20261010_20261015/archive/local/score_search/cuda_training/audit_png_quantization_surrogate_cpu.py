"""CPU-only bit and identity-gradient checks for the formal PNG surrogate."""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from png_quantization_surrogate import make_png_lut, png_quantization_ste


def main():
    torch.set_num_threads(2)
    random = np.random.default_rng(20261014)
    thresholds = (np.arange(255, dtype=np.float32) + np.float32(.5)) / np.float32(255)
    cases = {
        'random_0_1': random.uniform(0, 1, 200000).astype(np.float32),
        'all_255_half_thresholds_and_adjacent': np.concatenate([
            np.nextafter(thresholds, np.float32(-np.inf)), thresholds,
            np.nextafter(thresholds, np.float32(np.inf))]),
        'bounds_and_out_of_range': np.array([
            0, 1, np.nextafter(np.float32(0), np.float32(-np.inf)),
            np.nextafter(np.float32(0), np.float32(np.inf)),
            np.nextafter(np.float32(1), np.float32(-np.inf)),
            np.nextafter(np.float32(1), np.float32(np.inf)),
            -.1, 1.1, -1, 2, -1000000, 1000000,
            -np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32),
    }
    lut = make_png_lut('cpu')
    results = {}
    for name, values in cases.items():
        rgb = torch.from_numpy(values.copy()).requires_grad_(True)
        # Original Runner returns FP32 [-1,1]; original renderer uses NumPy.
        runner = (rgb.detach() * 2.0 - 1.0).clamp(-1.0, 1.0).numpy()
        encoded = ((runner * 0.5 + 0.5) * 255).round().astype(np.uint8)
        reference = encoded.astype(np.float32) / 255.0
        actual = png_quantization_ste(rgb, lut)
        different = int(np.count_nonzero(actual.detach().numpy().view(np.uint32) !=
                                        reference.view(np.uint32)))
        assert different == 0, (name, different)
        weights = torch.from_numpy(random.normal(size=values.shape).astype(np.float32))
        (actual * weights).sum().backward()
        gradient_bitexact = np.array_equal(rgb.grad.numpy().view(np.uint32),
                                          weights.numpy().view(np.uint32))
        assert gradient_bitexact, name
        fallback = png_quantization_ste(rgb.detach())
        assert torch.equal(fallback, actual.detach())
        results[name] = dict(pixel_count=values.size, forward_differing_bits_values=different,
                             identity_gradient_bitexact=True, optional_lut_same=True)
    p = Path(__file__).with_name('png_quantization_surrogate.py')
    report = dict(status='passed', device='cpu', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                  cuda_initialized=torch.cuda.is_initialized(), torch=torch.__version__, numpy=np.__version__,
                  helper_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), tests=results,
                  scope='Pure CPU eager FP32; no GPU or model initialization',
                  note='Identity gradient is an STE estimator, not the true derivative of round.')
    assert report['cuda_initialized'] is False
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
