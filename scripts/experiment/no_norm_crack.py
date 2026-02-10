"""No-Normalization Key Cracking.

The bottleneck is: angular_field /= max(|angular_field|)
This destroys ~50 bits of information.

Test: encode WITHOUT normalization, then crack.
If we get 256/256, the normalization was the only barrier.
Then explore alternative normalizations that preserve all bits.
"""

import sys
import time

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.utils.constants import NUM_THREADS

GRID_SIZE = 78


def build_basis(grid_size):
    """Build SH basis matrix."""
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    weight = (np.sin(theta_grid) * dtheta * dphi).ravel()

    basis = np.zeros((grid_size * grid_size, NUM_THREADS), dtype=np.float64)
    bit_idx = 0
    degree = 0
    while bit_idx < NUM_THREADS:
        for m in range(-degree, degree + 1):
            if bit_idx >= NUM_THREADS:
                break
            ylm = sph_harm_y(degree, m, theta_grid, phi_grid).real
            basis[:, bit_idx] = ylm.ravel()
            bit_idx += 1
        degree += 1

    return basis, weight


def encode_original(bits, basis):
    """Original encoding WITH normalization (the 80% ceiling)."""
    coeffs = 2.0 * np.array(bits, dtype=np.float64) - 1.0
    field = basis @ coeffs
    max_val = np.abs(field).max()
    if max_val > 0:
        field /= max_val
    return field


def encode_raw(bits, basis):
    """Raw encoding -- NO normalization."""
    coeffs = 2.0 * np.array(bits, dtype=np.float64) - 1.0
    field = basis @ coeffs
    return field


def encode_l2_norm(bits, basis):
    """L2-normalized encoding -- divide by L2 norm instead of max."""
    coeffs = 2.0 * np.array(bits, dtype=np.float64) - 1.0
    field = basis @ coeffs
    l2 = np.linalg.norm(field)
    if l2 > 0:
        field /= l2
    return field


def encode_softmax_norm(bits, basis):
    """Soft normalization -- divide by mean(|field|) instead of max."""
    coeffs = 2.0 * np.array(bits, dtype=np.float64) - 1.0
    field = basis @ coeffs
    mean_abs = np.mean(np.abs(field))
    if mean_abs > 0:
        field /= mean_abs
    return field


def encode_sign_only(bits, basis):
    """Sign-preserving encoding -- store only the sign of each grid point."""
    coeffs = 2.0 * np.array(bits, dtype=np.float64) - 1.0
    field = basis @ coeffs
    return np.sign(field)


def encode_tanh(bits, basis):
    """Tanh-compressed encoding -- soft clamp to [-1, 1]."""
    coeffs = 2.0 * np.array(bits, dtype=np.float64) - 1.0
    field = basis @ coeffs
    # Scale so typical values are in [-3, 3] range for tanh
    scale = np.std(field) * 2
    if scale > 0:
        field = np.tanh(field / scale)
    return field


def sh_readback(field, basis, weight):
    """SH inner-product readback."""
    coefficients = np.zeros(NUM_THREADS)
    for i in range(NUM_THREADS):
        coefficients[i] = np.sum(field * basis[:, i] * weight)
    bits = [1 if c > 0 else 0 for c in coefficients]
    return bits, coefficients


def lsq_readback(field, basis):
    """LSQ readback."""
    x, _, _, _ = np.linalg.lstsq(basis, field, rcond=None)
    bits = [1 if c > 0 else 0 for c in x]
    return bits, x


def score(extracted, target):
    matches = sum(a == b for a, b in zip(extracted, target))
    return matches, matches / 256


def test_encoding(name, field, basis, weight, target_bits):
    """Test both readback methods on an encoded field."""
    sh_bits, sh_coeffs = sh_readback(field, basis, weight)
    sh_match, sh_rate = score(sh_bits, target_bits)

    lsq_bits, lsq_coeffs = lsq_readback(field, basis)
    lsq_match, lsq_rate = score(lsq_bits, target_bits)

    best = max(sh_match, lsq_match)
    best_rate = max(sh_rate, lsq_rate)

    # Coefficient analysis
    abs_sh = np.abs(sh_coeffs)
    near_zero = np.sum(abs_sh < np.percentile(abs_sh, 20))

    print(f"  {name:25s}:  SH={sh_match:3d}/256  LSQ={lsq_match:3d}/256  "
          f"best={best:3d}/256 ({best_rate:.4f})  low_conf={near_zero}")

    return best, best_rate, sh_coeffs


def main():
    print()
    print("#" * 70)
    print("#  NO-NORMALIZATION KEY CRACKING")
    print("#  Testing encoding variants to break the 80% ceiling")
    print("#" * 70)

    # Run on 5 random keys to confirm results aren't key-specific
    num_keys = 5
    keys = [KeyInput.random() for _ in range(num_keys)]

    basis, weight = build_basis(GRID_SIZE)
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Basis: {basis.shape}")
    print(f"  Testing {num_keys} random keys\n")

    encodings = [
        ("Original (max-norm)", encode_original),
        ("Raw (no norm)", encode_raw),
        ("L2-normalized", encode_l2_norm),
        ("Mean-abs normalized", encode_softmax_norm),
        ("Sign-only", encode_sign_only),
        ("Tanh-compressed", encode_tanh),
    ]

    # Accumulate results
    summary = {name: [] for name, _ in encodings}

    for ki, key in enumerate(keys):
        target_bits = key.as_bits

        print(f"{'='*70}")
        print(f"  KEY {ki+1}/{num_keys}: {key.as_hex[:32]}...")
        print(f"{'='*70}")

        for name, encode_fn in encodings:
            field = encode_fn(target_bits, basis)
            best, rate, _ = test_encoding(name, field, basis, weight, target_bits)
            summary[name].append(best)

        print()

    # ================================================================
    # SUMMARY
    # ================================================================
    print("=" * 70)
    print("  ENCODING COMPARISON (averaged across all keys)")
    print("=" * 70)
    print()
    print(f"  {'Encoding':25s}  {'Mean':>6s}  {'Min':>5s}  {'Max':>5s}  {'Perfect?':>8s}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*8}")

    for name, _ in encodings:
        vals = summary[name]
        mean_v = np.mean(vals)
        min_v = min(vals)
        max_v = max(vals)
        perfect = "YES" if min_v == 256 else "no"
        print(f"  {name:25s}  {mean_v:5.1f}  {min_v:5d}  {max_v:5d}  {perfect:>8s}")

    # Highlight the winner
    best_encoding = max(summary.keys(), key=lambda k: np.mean(summary[k]))
    best_mean = np.mean(summary[best_encoding])
    best_min = min(summary[best_encoding])

    print(f"\n  BEST: {best_encoding}")
    print(f"  Mean: {best_mean:.1f}/256, Min: {best_min}/256")

    if best_min == 256:
        print(f"\n  *** 100% KEY RECOVERY ACHIEVED ***")
        print(f"  The '{best_encoding}' encoding preserves all 256 bits.")
        print(f"  The normalization was the ONLY barrier to perfect cracking.")
    elif best_mean > 230:
        print(f"\n  Major improvement over original (was ~200/256)")
    else:
        print(f"\n  No significant improvement found")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
