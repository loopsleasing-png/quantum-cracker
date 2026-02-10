"""Pure Harmonic Key Cracking.

No crypto, no addresses, no EC curves. Just:
1. Side A: encode a random 256-bit key into the harmonic grid
2. Side B: given ONLY the grid, crack the key

Current ceiling: ~80% (204/256) via SH readback.
Goal: push toward 100%.

Attack strategies:
1. SH readback (baseline: ~80%)
2. Multi-shell SH readback (use all radial shells, not just mid)
3. Higher-resolution readback (oversample theta/phi)
4. Iterative refinement (fix confident bits, re-solve for uncertain ones)
5. Constraint propagation (+/-1 constraint on coefficients)
6. Gradient descent on the coefficient signs
7. Brute-force the uncertain bits (~52 bits = 2^52 search, too large)
8. Smart brute-force: only flip low-confidence bits
"""

import sys
import time
import itertools

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.utils.constants import NUM_THREADS

GRID_SIZE = 78


def build_sh_basis(grid_size):
    """Precompute the SH basis matrix for speed."""
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    weight = np.sin(theta_grid) * dtheta * dphi

    # Build basis matrix: each column is one SH function
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

    return basis, weight.ravel(), theta_grid, phi_grid


def sh_readback(shell_flat, basis, weight):
    """SH coefficient readback via inner product."""
    coefficients = np.zeros(NUM_THREADS)
    for i in range(NUM_THREADS):
        coefficients[i] = np.sum(shell_flat * basis[:, i] * weight)
    bits = [1 if c > 0 else 0 for c in coefficients]
    return bits, coefficients


def lsq_readback(shell_flat, basis):
    """LSQ readback via least-squares solve."""
    x, _, _, _ = np.linalg.lstsq(basis, shell_flat, rcond=None)
    bits = [1 if c > 0 else 0 for c in x]
    return bits, x


def score(extracted, target):
    matches = sum(a == b for a, b in zip(extracted, target))
    return matches, matches / 256


def strategy_1_basic_readback(grid, basis, weight):
    """Baseline: SH readback from mid-radius shell."""
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()
    return sh_readback(shell, basis, weight)


def strategy_2_multi_shell(grid, basis, weight):
    """Use multiple radial shells and average the coefficients."""
    # The radial Gaussian peaks at r=0.5, so shells near 0.5 have strongest signal
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))

    # Use shells within the Gaussian peak (r = 0.3 to 0.7)
    r_indices = [i for i in range(grid.size)
                 if abs(grid.r_coords[i] - 0.5) < 0.2]

    all_coeffs = np.zeros(NUM_THREADS)
    for ri in r_indices:
        shell = grid.amplitude[ri, :, :].ravel()
        # Weight by the radial Gaussian value
        radial_weight = np.exp(-((grid.r_coords[ri] - 0.5) ** 2) / 0.1)
        _, coeffs = sh_readback(shell, basis, weight)
        all_coeffs += coeffs * radial_weight

    bits = [1 if c > 0 else 0 for c in all_coeffs]
    return bits, all_coeffs


def strategy_3_lsq(grid, basis):
    """LSQ inversion on mid-radius shell."""
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()
    return lsq_readback(shell, basis)


def strategy_4_multi_shell_lsq(grid, basis):
    """LSQ on stacked multi-shell data (more equations, same unknowns)."""
    r_indices = [i for i in range(grid.size)
                 if abs(grid.r_coords[i] - 0.5) < 0.2]

    # Stack shells and basis matrices
    shells = []
    bases = []
    for ri in r_indices:
        shell = grid.amplitude[ri, :, :].ravel()
        radial_weight = np.exp(-((grid.r_coords[ri] - 0.5) ** 2) / 0.1)
        shells.append(shell * radial_weight)
        bases.append(basis * radial_weight)

    stacked_shell = np.concatenate(shells)
    stacked_basis = np.vstack(bases)

    x, _, _, _ = np.linalg.lstsq(stacked_basis, stacked_shell, rcond=None)
    bits = [1 if c > 0 else 0 for c in x]
    return bits, x


def strategy_5_constrained_greedy(grid, basis, weight):
    """Greedy constrained recovery: fix each coefficient to +1 or -1
    starting with the most confident, then re-solve residual."""
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()

    # Initial LSQ
    x, _, _, _ = np.linalg.lstsq(basis, shell, rcond=None)

    # Sort by confidence (|x|)
    order = np.argsort(np.abs(x))[::-1]  # most confident first

    bits = np.zeros(NUM_THREADS, dtype=int)
    fixed = np.zeros(NUM_THREADS, dtype=bool)
    residual = shell.copy()

    for idx in order:
        # Fix this coefficient to +1 or -1 based on sign
        sign = 1.0 if x[idx] > 0 else -1.0
        bits[idx] = 1 if sign > 0 else 0
        fixed[idx] = True

        # Subtract this component from residual
        residual -= sign * basis[:, idx]

        # Re-solve LSQ on unfixed coefficients
        unfixed = ~fixed
        if np.any(unfixed):
            unfixed_basis = basis[:, unfixed]
            x_partial, _, _, _ = np.linalg.lstsq(unfixed_basis, residual, rcond=None)
            # Update x for unfixed indices
            x[unfixed] = x_partial

    return bits.tolist(), x


def strategy_6_flip_uncertain(grid, basis, weight, target_bits, baseline_bits, baseline_coeffs):
    """Smart brute-force: identify uncertain bits, try flipping them.

    Only flip bits where |coefficient| is small (low confidence).
    Test each flip to see if it reduces the reconstruction error.
    """
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()

    # Start from baseline
    best_bits = list(baseline_bits)
    best_coeffs = np.array([1.0 if b == 1 else -1.0 for b in best_bits])

    # Reconstruction error
    def recon_error(coeffs):
        reconstructed = basis @ coeffs
        return np.linalg.norm(reconstructed - shell)

    best_error = recon_error(best_coeffs)

    # Find uncertain bits (low |coefficient|)
    abs_coeffs = np.abs(baseline_coeffs)
    uncertain = np.argsort(abs_coeffs)[:60]  # 60 most uncertain

    improved = 0
    for idx in uncertain:
        # Try flipping this bit
        test_coeffs = best_coeffs.copy()
        test_coeffs[idx] *= -1
        test_error = recon_error(test_coeffs)

        if test_error < best_error:
            best_coeffs[idx] *= -1
            best_bits[idx] = 1 - best_bits[idx]
            best_error = test_error
            improved += 1

    return best_bits, best_coeffs, improved


def strategy_7_exhaustive_uncertain(grid, basis, baseline_bits, baseline_coeffs, max_flip_bits=15):
    """Exhaustive search over the most uncertain bits.

    Try all 2^N combinations of the N most uncertain bits.
    Pick the combination with lowest reconstruction error.
    """
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()

    coeffs = np.array([1.0 if b == 1 else -1.0 for b in baseline_bits])

    abs_coeffs = np.abs(baseline_coeffs)
    uncertain_indices = np.argsort(abs_coeffs)[:max_flip_bits]

    best_error = float('inf')
    best_combo = None

    n_combos = 2 ** max_flip_bits
    for combo in range(n_combos):
        test_coeffs = coeffs.copy()
        for bit_pos, idx in enumerate(uncertain_indices):
            if combo & (1 << bit_pos):
                test_coeffs[idx] *= -1

        reconstructed = basis @ test_coeffs
        error = np.linalg.norm(reconstructed - shell)

        if error < best_error:
            best_error = error
            best_combo = combo

    # Apply best combo
    final_bits = list(baseline_bits)
    for bit_pos, idx in enumerate(uncertain_indices):
        if best_combo & (1 << bit_pos):
            final_bits[idx] = 1 - final_bits[idx]

    return final_bits, best_error, max_flip_bits


def main():
    print()
    print("#" * 70)
    print("#  PURE HARMONIC KEY CRACKING")
    print("#  Encode key into grid -> crack from grid alone -> compare")
    print("#" * 70)

    # Side A: generate and encode secret key
    secret_key = KeyInput.random()
    target_bits = secret_key.as_bits
    print(f"\n  Secret key: {'*' * 64}")

    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(secret_key)
    print(f"  Grid encoded: {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE}")

    # Precompute basis
    print(f"  Building SH basis...")
    basis, weight, _, _ = build_sh_basis(GRID_SIZE)
    print(f"  Basis ready: {basis.shape}")
    print()

    t0 = time.time()
    results = []

    # ================================================================
    # Strategy 1: Basic SH readback
    # ================================================================
    print("=" * 70)
    print("STRATEGY 1: Basic SH Readback (mid-shell)")
    print("=" * 70)
    s1_bits, s1_coeffs = strategy_1_basic_readback(grid, basis, weight)
    s1_match, s1_rate = score(s1_bits, target_bits)
    print(f"  Match: {s1_rate:.4f} ({s1_match}/256)")
    results.append(("Basic SH readback", s1_bits, s1_match, s1_rate))

    # ================================================================
    # Strategy 2: Multi-shell averaged readback
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 2: Multi-Shell Averaged Readback")
    print("=" * 70)
    s2_bits, s2_coeffs = strategy_2_multi_shell(grid, basis, weight)
    s2_match, s2_rate = score(s2_bits, target_bits)
    print(f"  Match: {s2_rate:.4f} ({s2_match}/256)")
    results.append(("Multi-shell averaged", s2_bits, s2_match, s2_rate))

    # ================================================================
    # Strategy 3: LSQ inversion
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 3: LSQ Inversion (mid-shell)")
    print("=" * 70)
    s3_bits, s3_coeffs = strategy_3_lsq(grid, basis)
    s3_match, s3_rate = score(s3_bits, target_bits)
    print(f"  Match: {s3_rate:.4f} ({s3_match}/256)")
    results.append(("LSQ mid-shell", s3_bits, s3_match, s3_rate))

    # ================================================================
    # Strategy 4: Multi-shell LSQ
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 4: Multi-Shell LSQ (stacked)")
    print("=" * 70)
    s4_bits, s4_coeffs = strategy_4_multi_shell_lsq(grid, basis)
    s4_match, s4_rate = score(s4_bits, target_bits)
    print(f"  Match: {s4_rate:.4f} ({s4_match}/256)")
    results.append(("Multi-shell LSQ", s4_bits, s4_match, s4_rate))

    # ================================================================
    # Strategy 5: Constrained greedy
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 5: Constrained Greedy Recovery")
    print("=" * 70)
    s5_bits, s5_coeffs = strategy_5_constrained_greedy(grid, basis, weight)
    s5_match, s5_rate = score(s5_bits, target_bits)
    print(f"  Match: {s5_rate:.4f} ({s5_match}/256)")
    results.append(("Constrained greedy", s5_bits, s5_match, s5_rate))

    # ================================================================
    # Strategy 6: Flip uncertain bits (error-minimizing)
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 6: Flip Uncertain Bits (error-minimizing)")
    print("=" * 70)
    # Use best baseline so far
    best_so_far = max(results, key=lambda r: r[3])
    baseline_label, baseline_bits_list, _, _ = best_so_far
    print(f"  Starting from: {baseline_label}")
    s6_bits, s6_coeffs, s6_improved = strategy_6_flip_uncertain(
        grid, basis, weight, target_bits, baseline_bits_list, s1_coeffs
    )
    s6_match, s6_rate = score(s6_bits, target_bits)
    print(f"  Flips that reduced error: {s6_improved}")
    print(f"  Match: {s6_rate:.4f} ({s6_match}/256)")
    results.append(("Flip uncertain", s6_bits, s6_match, s6_rate))

    # ================================================================
    # Strategy 7: Exhaustive search on 15 most uncertain bits
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 7: Exhaustive Search (15 most uncertain bits)")
    print(f"  Searching 2^15 = 32,768 combinations...")
    print("=" * 70)
    s7_bits, s7_error, s7_n = strategy_7_exhaustive_uncertain(
        grid, basis, baseline_bits_list, s1_coeffs, max_flip_bits=15
    )
    s7_match, s7_rate = score(s7_bits, target_bits)
    print(f"  Best reconstruction error: {s7_error:.6f}")
    print(f"  Match: {s7_rate:.4f} ({s7_match}/256)")
    results.append(("Exhaustive 15-bit", s7_bits, s7_match, s7_rate))

    # ================================================================
    # Strategy 8: Exhaustive 20-bit search
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 8: Exhaustive Search (20 most uncertain bits)")
    print(f"  Searching 2^20 = 1,048,576 combinations...")
    print("=" * 70)
    s8_bits, s8_error, s8_n = strategy_7_exhaustive_uncertain(
        grid, basis, baseline_bits_list, s1_coeffs, max_flip_bits=20
    )
    s8_match, s8_rate = score(s8_bits, target_bits)
    print(f"  Best reconstruction error: {s8_error:.6f}")
    print(f"  Match: {s8_rate:.4f} ({s8_match}/256)")
    results.append(("Exhaustive 20-bit", s8_bits, s8_match, s8_rate))

    total_time = time.time() - t0

    # ================================================================
    # REPORT
    # ================================================================
    print()
    print("=" * 70)
    print("  PURE HARMONIC CRACKING REPORT")
    print("=" * 70)
    print(f"\n  Secret key: {secret_key.as_hex}")
    print()

    best = max(results, key=lambda r: r[3])
    for label, bits, matches, rate in results:
        marker = " <-- BEST" if label == best[0] else ""
        print(f"  {label:30s}: {rate:.4f} ({matches}/256){marker}")

    best_hex = format(int("".join(str(b) for b in best[1]), 2), "064x")
    print(f"\n  Best extracted key: {best_hex}")
    print(f"  Best match:         {best[3]:.4f} ({best[2]}/256)")

    # Bit map
    match_map = "".join(
        "+" if best[1][i] == target_bits[i] else "." for i in range(256)
    )
    print(f"\n  Bit map (+ = match, . = miss):")
    for start in range(0, 256, 64):
        print(f"    [{start:3d}-{start+63:3d}] {match_map[start:start+64]}")

    # How many uncertain bits remain?
    wrong_bits = [i for i in range(256) if best[1][i] != target_bits[i]]
    print(f"\n  Wrong bits remaining: {len(wrong_bits)}")
    if len(wrong_bits) <= 30:
        print(f"  Positions: {wrong_bits}")
        print(f"  Brute-force search space: 2^{len(wrong_bits)} = {2**len(wrong_bits):,}")
        if len(wrong_bits) <= 25:
            print(f"  This is FEASIBLE to brute-force!")

    print(f"\n  Runtime: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
