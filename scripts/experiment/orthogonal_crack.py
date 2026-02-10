"""Orthogonal Basis Key Cracking.

The SH basis is only approximately orthogonal on a discrete grid.
This creates ~50 ambiguous bit pairs that can't be resolved.

Fix: use a basis that IS perfectly orthogonal on the discrete grid.
Options:
1. QR-orthogonalized SH basis (Gram-Schmidt on the discrete grid)
2. Random orthogonal basis (guaranteed orthogonal)
3. Walsh-Hadamard basis (perfectly orthogonal, binary structure)
4. SVD-cleaned SH basis (remove near-degenerate modes)

If ANY of these give 256/256, we've proven the encoding can work
and the bottleneck was SH discretization error.
"""

import sys
import time

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.utils.constants import NUM_THREADS

GRID_SIZE = 78


def build_sh_basis(grid_size):
    """Standard SH basis (approximately orthogonal on discrete grid)."""
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


def orthogonalize_qr(basis):
    """QR orthogonalization: make columns exactly orthogonal on the grid."""
    Q, R = np.linalg.qr(basis)
    return Q


def orthogonalize_weighted_qr(basis, weight):
    """QR with integration weight: orthogonal w.r.t. weighted inner product."""
    # W^{1/2} * basis, then QR, then W^{-1/2} * Q
    sqrt_w = np.sqrt(weight)[:, np.newaxis]
    weighted_basis = basis * sqrt_w
    Q, R = np.linalg.qr(weighted_basis)
    # Q columns are orthonormal in the weighted sense
    return Q, weighted_basis


def random_orthogonal_basis(n_points, n_vectors):
    """Generate a random orthogonal basis via QR of random matrix."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((n_points, n_vectors))
    Q, _ = np.linalg.qr(A)
    return Q


def encode_and_crack(bits, basis, label):
    """Encode key into field using given basis, then crack via LSQ and inner product."""
    coeffs = 2.0 * np.array(bits, dtype=np.float64) - 1.0

    # Encode
    field = basis @ coeffs

    # Crack via LSQ
    x_lsq, _, _, sv = np.linalg.lstsq(basis, field, rcond=None)
    lsq_bits = [1 if c > 0 else 0 for c in x_lsq]
    lsq_match = sum(a == b for a, b in zip(lsq_bits, bits))

    # Crack via pseudo-inverse (should be exact if basis is orthogonal)
    x_pinv = np.linalg.pinv(basis) @ field
    pinv_bits = [1 if c > 0 else 0 for c in x_pinv]
    pinv_match = sum(a == b for a, b in zip(pinv_bits, bits))

    # Check orthogonality: basis^T @ basis should be diagonal
    gram = basis.T @ basis
    off_diag = gram - np.diag(np.diag(gram))
    max_off_diag = np.abs(off_diag).max()
    mean_off_diag = np.mean(np.abs(off_diag))

    # Condition number
    cond = float(sv[0] / sv[-1]) if len(sv) > 0 and sv[-1] > 0 else float('inf')

    # Coefficient analysis
    abs_lsq = np.abs(x_lsq)
    min_coeff = abs_lsq.min()
    mean_coeff = abs_lsq.mean()
    near_zero = np.sum(abs_lsq < 0.5)  # true coeffs are +/-1

    # Check if recovered coefficients are close to +/-1
    coeff_error = np.mean(np.abs(np.abs(x_lsq) - 1.0))

    best = max(lsq_match, pinv_match)
    print(f"  {label:35s}: LSQ={lsq_match:3d}  pinv={pinv_match:3d}  "
          f"best={best:3d}/256  cond={cond:.1e}  "
          f"off_diag={max_off_diag:.2e}  coeff_err={coeff_error:.4f}")

    return best, x_lsq, cond


def encode_with_norm_and_crack(bits, basis, label):
    """Encode WITH max-normalization, then crack."""
    coeffs = 2.0 * np.array(bits, dtype=np.float64) - 1.0
    field = basis @ coeffs
    max_val = np.abs(field).max()
    if max_val > 0:
        field /= max_val

    x_lsq, _, _, sv = np.linalg.lstsq(basis, field, rcond=None)
    lsq_bits = [1 if c > 0 else 0 for c in x_lsq]
    lsq_match = sum(a == b for a, b in zip(lsq_bits, bits))

    x_pinv = np.linalg.pinv(basis) @ field
    pinv_bits = [1 if c > 0 else 0 for c in x_pinv]
    pinv_match = sum(a == b for a, b in zip(pinv_bits, bits))

    cond = float(sv[0] / sv[-1]) if len(sv) > 0 and sv[-1] > 0 else float('inf')
    best = max(lsq_match, pinv_match)

    print(f"  {label:35s}: LSQ={lsq_match:3d}  pinv={pinv_match:3d}  "
          f"best={best:3d}/256  cond={cond:.1e}  (with max-norm)")

    return best


def main():
    print()
    print("#" * 70)
    print("#  ORTHOGONAL BASIS KEY CRACKING")
    print("#  Fix the SH discretization problem with exact orthogonal bases")
    print("#" * 70)

    num_keys = 5
    keys = [KeyInput.random() for _ in range(num_keys)]
    n_pts = GRID_SIZE * GRID_SIZE

    # Build bases
    print(f"\n  Building bases ({GRID_SIZE}x{GRID_SIZE} = {n_pts} points)...")

    t0 = time.time()
    sh_basis, weight = build_sh_basis(GRID_SIZE)
    sh_qr = orthogonalize_qr(sh_basis)
    qr_weighted, _ = orthogonalize_weighted_qr(sh_basis, weight)
    rand_basis = random_orthogonal_basis(n_pts, NUM_THREADS)
    print(f"  Bases built in {time.time()-t0:.1f}s\n")

    bases = [
        ("SH basis (original)", sh_basis),
        ("SH + QR orthogonalized", sh_qr),
        ("SH + weighted QR", qr_weighted),
        ("Random orthogonal", rand_basis),
    ]

    # Track results
    summary = {name: [] for name, _ in bases}
    summary_norm = {name: [] for name, _ in bases}

    for ki, key in enumerate(keys):
        target_bits = key.as_bits

        print(f"{'='*70}")
        print(f"  KEY {ki+1}/{num_keys}: {key.as_hex[:32]}...")
        print(f"{'='*70}")

        print(f"\n  WITHOUT normalization:")
        for name, basis in bases:
            best, _, _ = encode_and_crack(target_bits, basis, name)
            summary[name].append(best)

        print(f"\n  WITH max-normalization:")
        for name, basis in bases:
            best = encode_with_norm_and_crack(target_bits, basis, name)
            summary_norm[name].append(best)

        print()

    # ================================================================
    # SUMMARY
    # ================================================================
    print("=" * 70)
    print("  RESULTS: WITHOUT NORMALIZATION")
    print("=" * 70)
    print(f"\n  {'Basis':35s}  {'Mean':>6s}  {'Min':>5s}  {'Max':>5s}  {'100%?':>5s}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    for name, _ in bases:
        vals = summary[name]
        mean_v = np.mean(vals)
        perfect = "YES" if min(vals) == 256 else "no"
        print(f"  {name:35s}  {mean_v:5.1f}  {min(vals):5d}  {max(vals):5d}  {perfect:>5s}")

    print()
    print("=" * 70)
    print("  RESULTS: WITH MAX-NORMALIZATION")
    print("=" * 70)
    print(f"\n  {'Basis':35s}  {'Mean':>6s}  {'Min':>5s}  {'Max':>5s}  {'100%?':>5s}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    for name, _ in bases:
        vals = summary_norm[name]
        mean_v = np.mean(vals)
        perfect = "YES" if min(vals) == 256 else "no"
        print(f"  {name:35s}  {mean_v:5.1f}  {min(vals):5d}  {max(vals):5d}  {perfect:>5s}")

    # Highlight
    print()
    for name, _ in bases:
        if min(summary[name]) == 256:
            print(f"  *** {name}: 256/256 on ALL keys (without norm) ***")
        if min(summary_norm[name]) == 256:
            print(f"  *** {name}: 256/256 on ALL keys (WITH norm) ***")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
