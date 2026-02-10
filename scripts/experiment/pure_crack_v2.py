"""Pure Harmonic Key Cracking v2 -- Attack the Normalization.

The encoding does: angular_field /= max(|angular_field|)
This normalization loses information. But we know:
1. Each coefficient is exactly +1 or -1 (before normalization)
2. The basis functions are known (SH modes)
3. The normalization scalar is max(|sum of +/-1 * Y_lm|)

Strategy: solve for the normalization constant, then recover exact +/-1.
Also: use the grid encoding process itself to build an exact inverse.
"""

import sys
import time

import numpy as np
from scipy.special import sph_harm_y
from scipy.optimize import minimize

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.utils.constants import NUM_THREADS

GRID_SIZE = 78


def build_sh_basis(grid_size):
    """Precompute SH basis on the angular grid."""
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

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

    return basis, theta_grid, phi_grid


def encode_key_to_field(bits, basis, grid_size):
    """Replicate the exact encoding from KeyInput.to_grid_state().

    Returns the normalized angular field (what's stored in the grid).
    """
    coeffs = 2.0 * np.array(bits, dtype=np.float64) - 1.0  # 0->-1, 1->+1
    angular_field = basis @ coeffs  # shape (grid_size^2,)

    max_val = np.abs(angular_field).max()
    if max_val > 0:
        angular_field /= max_val

    return angular_field, max_val


def score(extracted, target):
    matches = sum(a == b for a, b in zip(extracted, target))
    return matches, matches / 256


def strategy_exact_inverse(grid, basis, grid_size):
    """Try to exactly invert the encoding.

    The encoding is: field = normalize(basis @ coeffs)
    where coeffs are +/-1 and normalize divides by max(|field|).

    If we know the normalization constant alpha:
      field * alpha = basis @ coeffs
      coeffs = basis^{-1} @ (field * alpha)

    We try many alpha values and check which gives coeffs closest to +/-1.
    """
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()

    # The encoding multiplied by radial Gaussian at r=0.5
    radial_weight = np.exp(-((grid.r_coords[r_mid] - 0.5) ** 2) / 0.1)
    # shell = radial_weight * normalized_angular_field
    # So: normalized_angular_field = shell / radial_weight
    angular_field = shell / radial_weight

    # LSQ to get raw coefficients (these are normalized)
    x_norm, _, _, _ = np.linalg.lstsq(basis, angular_field, rcond=None)

    # The true coefficients are +/-1. After normalization by alpha:
    # x_norm = true_coeffs / alpha
    # So: true_coeffs = x_norm * alpha
    # And |true_coeffs| should all be 1.0

    # If we scale x_norm so that the median |x| = 1, we get alpha
    median_abs = np.median(np.abs(x_norm))
    alpha_estimate = 1.0 / median_abs if median_abs > 0 else 1.0

    # Try range of alpha values around the estimate
    best_bits = None
    best_score = 0
    best_alpha = 0

    for alpha_mult in np.linspace(0.5, 2.0, 200):
        alpha = alpha_estimate * alpha_mult
        true_coeffs = x_norm * alpha
        bits = [1 if c > 0 else 0 for c in true_coeffs]

        # Score: how close are |coeffs| to 1.0?
        deviation = np.mean(np.abs(np.abs(true_coeffs) - 1.0))

        # Also check reconstruction error
        reconstructed = basis @ (2.0 * np.array(bits, dtype=float) - 1.0)
        max_recon = np.abs(reconstructed).max()
        if max_recon > 0:
            reconstructed /= max_recon
        recon_error = np.linalg.norm(reconstructed * radial_weight - shell)

        # Combined score (lower is better)
        combined = deviation + recon_error * 0.01

        if best_bits is None or combined < best_score:
            best_score = combined
            best_bits = bits
            best_alpha = alpha

    return best_bits, x_norm, best_alpha


def strategy_reconstruction_match(grid, basis, grid_size):
    """For each possible bit pattern, check if its encoding matches the grid.

    Since we can't try all 2^256, we use a greedy approach:
    1. Start from LSQ solution
    2. For each bit, check if flipping it makes the reconstruction closer
    3. Iterate until no more improvements
    """
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()
    radial_weight = np.exp(-((grid.r_coords[r_mid] - 0.5) ** 2) / 0.1)

    # Start from LSQ
    x, _, _, _ = np.linalg.lstsq(basis, shell / radial_weight, rcond=None)
    bits = np.array([1 if c > 0 else 0 for c in x])

    def recon_error(bits_arr):
        coeffs = 2.0 * bits_arr.astype(float) - 1.0
        field = basis @ coeffs
        max_val = np.abs(field).max()
        if max_val > 0:
            field /= max_val
        return np.linalg.norm(field * radial_weight - shell)

    best_error = recon_error(bits)
    improved = True
    total_flips = 0
    iteration = 0

    while improved:
        improved = False
        iteration += 1
        for i in range(256):
            bits[i] = 1 - bits[i]  # flip
            new_error = recon_error(bits)
            if new_error < best_error:
                best_error = new_error
                improved = True
                total_flips += 1
            else:
                bits[i] = 1 - bits[i]  # flip back

    return bits.tolist(), best_error, total_flips, iteration


def strategy_simulated_annealing(grid, basis, grid_size, max_iter=50000):
    """Simulated annealing on the bit pattern.

    Minimize reconstruction error by randomly flipping bits,
    accepting worse solutions with decreasing probability.
    """
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()
    radial_weight = np.exp(-((grid.r_coords[r_mid] - 0.5) ** 2) / 0.1)

    # Start from LSQ
    x, _, _, _ = np.linalg.lstsq(basis, shell / radial_weight, rcond=None)
    bits = np.array([1 if c > 0 else 0 for c in x])

    # Precompute for speed
    def recon_error(bits_arr):
        coeffs = 2.0 * bits_arr.astype(float) - 1.0
        field = basis @ coeffs
        max_val = np.abs(field).max()
        if max_val > 0:
            field /= max_val
        return np.linalg.norm(field * radial_weight - shell)

    current_error = recon_error(bits)
    best_bits = bits.copy()
    best_error = current_error

    rng = np.random.default_rng(42)

    for step in range(max_iter):
        T = 1.0 * (1.0 - step / max_iter)  # linear cooling

        # Random bit flip
        idx = rng.integers(256)
        bits[idx] = 1 - bits[idx]
        new_error = recon_error(bits)

        delta = new_error - current_error
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / (T + 1e-10))):
            current_error = new_error
            if current_error < best_error:
                best_error = current_error
                best_bits = bits.copy()
        else:
            bits[idx] = 1 - bits[idx]  # reject

    return best_bits.tolist(), best_error


def main():
    print()
    print("#" * 70)
    print("#  PURE HARMONIC KEY CRACKING v2")
    print("#  Attacking the normalization to break the 80% ceiling")
    print("#" * 70)

    secret_key = KeyInput.random()
    target_bits = secret_key.as_bits
    print(f"\n  Secret key: {'*' * 64}")

    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(secret_key)

    basis, _, _ = build_sh_basis(GRID_SIZE)
    print(f"  Basis: {basis.shape}")

    t0 = time.time()
    results = []

    # ================================================================
    # Strategy 1: Exact inverse (alpha recovery)
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 1: Exact Inverse (normalization constant recovery)")
    print("=" * 70)
    s1_bits, s1_coeffs, s1_alpha = strategy_exact_inverse(grid, basis, GRID_SIZE)
    s1_match, s1_rate = score(s1_bits, target_bits)
    print(f"  Estimated alpha: {s1_alpha:.6f}")
    print(f"  Match: {s1_rate:.4f} ({s1_match}/256)")
    results.append(("Exact inverse", s1_bits, s1_match, s1_rate))

    # ================================================================
    # Strategy 2: Reconstruction matching (greedy)
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 2: Reconstruction Matching (greedy bit flips)")
    print("=" * 70)
    s2_bits, s2_error, s2_flips, s2_iters = strategy_reconstruction_match(
        grid, basis, GRID_SIZE
    )
    s2_match, s2_rate = score(s2_bits, target_bits)
    print(f"  Flips made: {s2_flips}, iterations: {s2_iters}")
    print(f"  Final recon error: {s2_error:.6f}")
    print(f"  Match: {s2_rate:.4f} ({s2_match}/256)")
    results.append(("Recon matching", s2_bits, s2_match, s2_rate))

    # ================================================================
    # Strategy 3: Simulated annealing
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 3: Simulated Annealing (50,000 iterations)")
    print("=" * 70)
    s3_bits, s3_error = strategy_simulated_annealing(grid, basis, GRID_SIZE)
    s3_match, s3_rate = score(s3_bits, target_bits)
    print(f"  Final recon error: {s3_error:.6f}")
    print(f"  Match: {s3_rate:.4f} ({s3_match}/256)")
    results.append(("Simulated annealing", s3_bits, s3_match, s3_rate))

    # ================================================================
    # Strategy 4: SA with more iterations
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 4: Simulated Annealing (200,000 iterations)")
    print("=" * 70)
    s4_bits, s4_error = strategy_simulated_annealing(grid, basis, GRID_SIZE, max_iter=200000)
    s4_match, s4_rate = score(s4_bits, target_bits)
    print(f"  Final recon error: {s4_error:.6f}")
    print(f"  Match: {s4_rate:.4f} ({s4_match}/256)")
    results.append(("SA 200K iters", s4_bits, s4_match, s4_rate))

    total_time = time.time() - t0

    # ================================================================
    # REPORT
    # ================================================================
    print()
    print("=" * 70)
    print("  PURE HARMONIC CRACKING v2 REPORT")
    print("=" * 70)
    print(f"\n  Secret key: {secret_key.as_hex}")
    print()

    best = max(results, key=lambda r: r[3])
    for label, bits, matches, rate in results:
        marker = " <-- BEST" if label == best[0] else ""
        print(f"  {label:30s}: {rate:.4f} ({matches}/256){marker}")

    best_hex = format(int("".join(str(b) for b in best[1]), 2), "064x")
    print(f"\n  Best extracted: {best_hex}")
    print(f"  Target:         {secret_key.as_hex}")
    print(f"  Match:          {best[2]}/256")

    wrong = [i for i in range(256) if best[1][i] != target_bits[i]]
    print(f"\n  Wrong bits: {len(wrong)}")
    if len(wrong) <= 30:
        print(f"  Positions: {wrong}")
        print(f"  Brute-force: 2^{len(wrong)} = {2**len(wrong):,}")

    print(f"\n  Runtime: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
