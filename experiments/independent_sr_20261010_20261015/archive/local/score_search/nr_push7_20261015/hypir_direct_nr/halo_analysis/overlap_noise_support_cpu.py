"""Small stdlib CPU toy of HYPIR tile overlap, not a network/metric evaluation.

Uses the installed Gaussian formula in Float64 for analytic conditional variance.
The dense per-tile mean fixture demonstrates support only, not a HYPIR Jacobian.
No torch import, GPU, model weights, output images, or training implementation.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import random

TILE = 64
STRIDE = 32


def origins(n):
    result = list(range(0, n - TILE + 1, STRIDE))
    if (n - TILE) % STRIDE:
        result.append(n - TILE)
    return result


def gaussian(i, axis):
    midpoint = (TILE - 1) / 2 if axis == 'x' else TILE / 2
    return math.exp(-(i - midpoint) ** 2 / TILE ** 2 / .02) / math.sqrt(2 * math.pi * .01)


def normalized_weights(n, p, axis):
    values = [(o, gaussian(p - o, axis)) for o in origins(n) if o <= p < o + TILE]
    total = sum(v for _, v in values)
    return [(o, v / total) for o, v in values]


def axis_variance(n, p, axis):
    return sum(w * w for _, w in normalized_weights(n, p, axis))


@lru_cache(None)
def tile_mean_row(n, p):
    row = {}
    for o, w in normalized_weights(n, p, 'x'):
        for q in range(o, o + TILE):
            row[q] = row.get(q, 0.) + w / TILE
    return row


@lru_cache(None)
def composed_row(n, p, stages):
    if stages == 1:
        return tile_mean_row(n, p)
    out = {}
    for q, weight in tile_mean_row(n, p).items():
        for r, v in composed_row(n, q, stages - 1).items():
            out[r] = out.get(r, 0.) + weight * v
    return out


def closure_fixture(n):
    # Context origins and the larger image use the same global 256px grid.
    global_origin = 128
    core = (n - TILE) // 2
    positions = sorted(set([core, core + 31, core + 63]))
    wavelet_positions = [core - 4, core + 67]
    results = []
    for p in positions + [p for p in wavelet_positions if 0 <= p < n]:
        local = {k + global_origin: v for k, v in composed_row(n, p, 3).items()}
        full = composed_row(512, p + global_origin, 3)
        outside = {k: v for k, v in full.items() if not global_origin <= k < global_origin + n}
        results.append(dict(local_latent_coordinate=p, scope='core' if p in positions else 'wavelet_radius31_ceil_to4_latent_support',
                            missing_nonzero_input_coordinates=len(outside),
                            missing_positive_linear_fixture_mass=sum(outside.values()),
                            linear_fixture_row_L1_difference=sum(abs(local.get(k, 0.) - full.get(k, 0.)) for k in set(local) | set(full))))
    return dict(context_pixels=n * 8, core_pixels=512, centered_halo_pixels=(n - TILE) * 4,
                context_origin_on_global_grid=True, core_origin_latent_phase=core % STRIDE, samples=results)


def main():
    out = Path(__file__).with_name('overlap_noise_support_cpu.json')
    assert not out.exists()
    contexts = []
    for n in [64, 96, 128, 192, 256, 320]:
        x = [axis_variance(n, p, 'x') for p in range(n)]
        y = [axis_variance(n, p, 'y') for p in range(n)]
        lo = (n - TILE) // 2
        contexts.append(dict(context_pixels=n * 8, encoder_tiles=len(origins(n)) ** 2,
                             image_mean_equal_sigma_variance_ratio=sum(x) * sum(y) / n ** 2,
                             center512_mean_equal_sigma_variance_ratio=sum(x[lo:lo + TILE]) * sum(y[lo:lo + TILE]) / TILE ** 2,
                             minimum_equal_sigma_variance_ratio=min(x) * min(y), maximum_equal_sigma_variance_ratio=max(x) * max(y)))
    phase_x = [axis_variance(512, 192 + p, 'x') for p in range(STRIDE)]
    phase_y = [axis_variance(512, 192 + p, 'y') for p in range(STRIDE)]
    equal_sigma_interior = sum(phase_x) * sum(phase_y) / STRIDE ** 2
    # Empirical independent N(0,1) draws validate the variance identity at one
    # crossover. This is Python RNG, not a replay of CUDA posterior draws.
    weights = [wx * wy for _, wx in normalized_weights(512, 208, 'x')
               for _, wy in normalized_weights(512, 208, 'y')]
    rng = random.Random(20261015)
    samples = [sum(w * rng.gauss(0., 1.) for w in weights) for _ in range(20000)]
    mean = sum(samples) / len(samples)
    empirical = sum((v - mean) ** 2 for v in samples) / (len(samples) - 1)
    result = dict(status='CPU_TOY_COMPLETE', CPU_only=True, torch_imported=False, GPU_used=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  actual_common_source_sha256='76dc674d6e0574c64f7522d66658927e50c7767cd27ead2fa594f7d996fbe923',
                  actual_base_source_sha256='99f5f91f49ef4248db9ba1dd4332ef0e01b5ecf6e729a267e40541cdce853161',
                  analytic_model='conditional independent tile samples with equal unit posterior sigma; Float64 original Gaussian formula; excludes actual means/sigmas and BF16 rounding',
                  variance_identity='Var(sum_i alpha_i*(mu_i+sigma_i*epsilon_i)|input)=sum_i alpha_i^2*sigma_i^2',
                  contexts=contexts, native_interior_phase_mean_equal_sigma_variance_ratio=equal_sigma_interior,
                  crossover_monte_carlo=dict(independent_samples=20000, contributing_tiles=len(weights), analytic_variance=sum(w*w for w in weights), empirical_variance=empirical, empirical_mean=mean),
                  support_fixture='one-dimensional three-stage Gaussian blend of full-tile means, latent64/stride32; positive fixture proves possible nonlocal dependencies, does not measure HYPIR Jacobian or error',
                  support= [closure_fixture(n) for n in [64, 96, 128, 192, 256, 320]],
                  wavelet=dict(levels=5, pixel_radius=sum(2 ** i for i in range(5)), latent_support_round_up=4,
                               strict_aligned_interior_pre_wavelet_halo_pixels=768, strict_aligned_interior_post_wavelet_halo_pixels=1024),
                  limitations=['No actual VAE posterior statistics, GPU BF16 arithmetic, quality metric, speed, or memory measurements.',
                               'Matching support does not match native tile draw indices or random-number consumption; no bitwise-equivalence claim.',
                               'Native end-anchored off-grid border tiles are not represented by the aligned interior fixture.',
                               'GroupNorm and attention permit whole-tile dependency; the uniform mean fixture is only a constructive support example.'])
    with out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(), interior_variance_ratio=equal_sigma_interior, crossover=result['crossover_monte_carlo'], context_support=[dict(context_pixels=r['context_pixels'], core_missing=max(s['missing_nonzero_input_coordinates'] for s in r['samples'] if s['scope']=='core'), extended_missing=max((s['missing_nonzero_input_coordinates'] for s in r['samples'] if s['scope']!='core'),default=None)) for r in result['support']]), indent=2))


if __name__ == '__main__':
    main()
