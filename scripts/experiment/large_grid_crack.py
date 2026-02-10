"""Large Grid Key Cracking.

The 78x78 grid has 6,084 angular points for 256 coefficients.
The normalization step creates degeneracy -- multiple +/-1 assignments
map to the same normalized field.

Hypothesis: a larger grid provides more angular resolution, reducing
the degeneracy and allowing more bits to be recovered.

Test grid sizes: 78, 156, 234, 312, 468, 624
"""

import sys
import time

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.utils.constants import NUM_THREADS


def build_basis_and_encode(bits_array, grid_size):
    """Build SH basis at given grid size and encode key bits.

    Returns: (angular_field_normalized, basis_matrix, radial_weight_at_mid)
    This replicates KeyInput.to_grid_state() but at arbitrary resolution.
    """
    coeffs = 2.0 * bits_array - 1.0  # 0->-1, 1->+1

    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Build basis matrix
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

    # Encode
    angular_field = basis @ coeffs
    max_val = np.abs(angular_field).max()
    norm_factor = max_val if max_val > 0 else 1.0
    angular_field_normalized = angular_field / norm_factor

    # Integration weight for readback
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    weight = (np.sin(theta_grid) * dtheta * dphi).ravel()

    return angular_field_normalized, basis, weight, norm_factor


def sh_readback(field, basis, weight):
    """Inner-product SH readback."""
    coefficients = np.zeros(NUM_THREADS)
    for i in range(NUM_THREADS):
        coefficients[i] = np.sum(field * basis[:, i] * weight)
    bits = [1 if c > 0 else 0 for c in coefficients]
    return bits, coefficients


def lsq_readback(field, basis):
    """LSQ readback."""
    x, _, _, sv = np.linalg.lstsq(basis, field, rcond=None)
    cond = float(sv[0] / sv[-1]) if len(sv) > 0 and sv[-1] > 0 else float('inf')
    bits = [1 if c > 0 else 0 for c in x]
    return bits, x, cond


def greedy_recon_match(field, basis):
    """Greedy reconstruction matching -- flip bits to minimize error."""
    x, _, _, _ = np.linalg.lstsq(basis, field, rcond=None)
    bits = np.array([1 if c > 0 else 0 for c in x])

    def recon_error(b):
        c = 2.0 * b.astype(float) - 1.0
        f = basis @ c
        mx = np.abs(f).max()
        if mx > 0:
            f /= mx
        return np.linalg.norm(f - field)

    best_error = recon_error(bits)
    improved = True
    flips = 0

    while improved:
        improved = False
        for i in range(256):
            bits[i] = 1 - bits[i]
            e = recon_error(bits)
            if e < best_error:
                best_error = e
                improved = True
                flips += 1
            else:
                bits[i] = 1 - bits[i]

    return bits.tolist(), best_error, flips


def score(extracted, target):
    matches = sum(a == b for a, b in zip(extracted, target))
    return matches, matches / 256


def test_grid_size(grid_size, target_bits, bits_array):
    """Run all strategies at a given grid size."""
    print(f"\n  Building {grid_size}x{grid_size} basis ({grid_size**2:,} angular points)...")
    t0 = time.time()

    field, basis, weight, norm_factor = build_basis_and_encode(bits_array, grid_size)
    build_time = time.time() - t0
    print(f"  Basis built in {build_time:.1f}s, norm_factor={norm_factor:.4f}")

    results = {}

    # SH readback
    t1 = time.time()
    sh_bits, sh_coeffs = sh_readback(field, basis, weight)
    sh_match, sh_rate = score(sh_bits, target_bits)
    results["sh_readback"] = (sh_match, sh_rate, time.time() - t1)
    print(f"  SH readback:     {sh_rate:.4f} ({sh_match}/256) [{time.time()-t1:.1f}s]")

    # LSQ readback
    t2 = time.time()
    lsq_bits, lsq_coeffs, cond = lsq_readback(field, basis)
    lsq_match, lsq_rate = score(lsq_bits, target_bits)
    results["lsq"] = (lsq_match, lsq_rate, time.time() - t2)
    print(f"  LSQ readback:    {lsq_rate:.4f} ({lsq_match}/256) [cond={cond:.2e}, {time.time()-t2:.1f}s]")

    # Greedy reconstruction matching
    t3 = time.time()
    greedy_bits, greedy_error, greedy_flips = greedy_recon_match(field, basis)
    greedy_match, greedy_rate = score(greedy_bits, target_bits)
    results["greedy"] = (greedy_match, greedy_rate, time.time() - t3)
    print(f"  Greedy recon:    {greedy_rate:.4f} ({greedy_match}/256) [error={greedy_error:.6f}, flips={greedy_flips}, {time.time()-t3:.1f}s]")

    # Confidence analysis
    abs_coeffs = np.abs(sh_coeffs)
    low_conf = np.sum(abs_coeffs < np.percentile(abs_coeffs, 20))
    print(f"  Low-confidence bits (<20th pctile): {low_conf}")

    total = time.time() - t0
    results["total_time"] = total
    results["best_match"] = max(sh_match, lsq_match, greedy_match)
    results["best_rate"] = max(sh_rate, lsq_rate, greedy_rate)
    results["condition_number"] = cond
    results["norm_factor"] = norm_factor

    return results


def main():
    print()
    print("#" * 70)
    print("#  LARGE GRID KEY CRACKING")
    print("#  Testing if higher angular resolution breaks the 80% ceiling")
    print("#" * 70)

    secret_key = KeyInput.random()
    target_bits = secret_key.as_bits
    bits_array = np.array(target_bits, dtype=np.float64)

    print(f"\n  Secret key: {secret_key.as_hex}")
    print(f"  Ones count: {sum(target_bits)}/256")

    grid_sizes = [78, 156, 234, 312, 468]

    all_results = {}

    for gs in grid_sizes:
        print()
        print("=" * 70)
        print(f"  GRID SIZE: {gs}x{gs} ({gs**2:,} angular points, {gs**2/256:.0f}x overdetermined)")
        print("=" * 70)

        try:
            results = test_grid_size(gs, target_bits, bits_array)
            all_results[gs] = results
        except MemoryError:
            print(f"  OUT OF MEMORY -- skipping")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            break

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("=" * 70)
    print("  GRID SIZE COMPARISON")
    print("=" * 70)
    print(f"\n  {'Grid':>6s}  {'Points':>8s}  {'Ratio':>6s}  {'SH':>8s}  {'LSQ':>8s}  {'Greedy':>8s}  {'Best':>8s}  {'Cond#':>10s}  {'Time':>6s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}")

    for gs in grid_sizes:
        if gs not in all_results:
            break
        r = all_results[gs]
        sh_m = r["sh_readback"][0]
        lsq_m = r["lsq"][0]
        gr_m = r["greedy"][0]
        best = r["best_match"]
        cond = r["condition_number"]
        t = r["total_time"]
        print(f"  {gs:>4d}^2  {gs**2:>8,}  {gs**2/256:>5.0f}x  "
              f"{sh_m:>5d}/256  {lsq_m:>5d}/256  {gr_m:>5d}/256  {best:>5d}/256  "
              f"{cond:>10.2e}  {t:>5.1f}s")

    # Did we break through?
    if all_results:
        best_overall = max(all_results.values(), key=lambda r: r["best_rate"])
        best_gs = [gs for gs, r in all_results.items() if r["best_rate"] == best_overall["best_rate"]][0]
        print(f"\n  Best result: {best_overall['best_match']}/256 ({best_overall['best_rate']:.4f}) at grid {best_gs}x{best_gs}")

        if best_overall["best_match"] > 220:
            print(f"  BREAKTHROUGH: {best_overall['best_match']}/256 exceeds previous ceiling!")
            wrong = 256 - best_overall["best_match"]
            print(f"  Only {wrong} bits wrong -- brute-force 2^{wrong} is {'feasible' if wrong <= 25 else 'not feasible'}")
        elif best_overall["best_match"] > 206:
            print(f"  IMPROVEMENT over 78x78 baseline (was 206/256)")
        else:
            print(f"  NO IMPROVEMENT -- the ceiling is structural, not resolution-limited")
            print(f"  The normalization step (dividing by max) is the bottleneck,")
            print(f"  not angular sampling resolution.")

    print()
    print("=" * 70)
    print(f"  Key: {secret_key.as_hex}")
    print("=" * 70)


if __name__ == "__main__":
    main()
