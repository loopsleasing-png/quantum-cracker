"""Phase 5: Resonance-Tuned Oracle (The "Missing Step" Experiment).

The user's intuition: "we are on to something with the frequency."

Instead of scoring raw EC remainder coordinates with mathematical
oracles, first PROCESS the remainder through the harmonic pipeline
at varying resonance frequencies. The harmonic system might act as
a lens that reveals structure invisible in raw coordinates.

For each bit position:
  1. Compute R_0 = K (bit=0) and R_1 = K - 2^i * G (bit=1)
  2. Feed each remainder's x-coord through harmonic pipeline
  3. Apply resonance at 9 different frequencies
  4. Score: which candidate has more concentrated energy?
"""

import csv
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import INFINITY, Point

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.utils.math_helpers import build_qr_sh_basis

G = SECP256k1.generator
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()

GRID_SIZE = 20
N_MODES = 256
FREQUENCIES = [1, 2, 5, 10, 20, 50, 78, 100, 156]
RESONANCE_STEPS = 10
RESONANCE_STRENGTH = 0.05

TARGET_HEX = "06d88f2148757a251dd0ea0e6c4584e159a60cfd3f7217c7b0b111adec0efbca"


def apply_resonance(grid_2d, freq, steps=RESONANCE_STEPS, strength=RESONANCE_STRENGTH):
    """Apply resonance at given frequency to a 2D angular grid.

    Simulates the harmonic compiler's vibration field.

    Args:
        grid_2d: (grid_size, grid_size) amplitude array
        freq: resonance frequency in MHz
        steps: number of time steps
        strength: vibration amplitude

    Returns:
        Modified grid after resonance
    """
    gs = grid_2d.shape[0]
    theta = np.linspace(0, np.pi, gs)
    phi = np.linspace(0, 2 * np.pi, gs)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    result = grid_2d.copy()
    for t in range(steps):
        vibration = np.sin(freq * phi_grid + t * 0.1) * np.cos(freq * theta_grid)
        result *= (1.0 + vibration * strength)

    return result


def key_to_resonated_coeffs(key_int, freq):
    """Convert a 256-bit integer to resonated SH coefficients.

    Pipeline: int -> KeyInput -> grid_state -> resonance -> SH readback
    """
    # Clamp to 256 bits
    key_hex = f"{key_int % (2**256):064x}"
    key = KeyInput(key_hex)
    bits = np.array(key.as_bits, dtype=np.float64)
    coeffs = 2.0 * bits - 1.0

    # Build angular field
    basis = build_qr_sh_basis(GRID_SIZE, N_MODES)
    n_points = GRID_SIZE * GRID_SIZE

    if n_points < N_MODES:
        return coeffs

    angular_field = (basis @ coeffs).reshape(GRID_SIZE, GRID_SIZE)

    # Normalize
    max_val = np.abs(angular_field).max()
    if max_val > 0:
        angular_field /= max_val

    # Apply resonance
    resonated = apply_resonance(angular_field, freq)

    # Read back SH coefficients
    readback_coeffs = basis.T @ resonated.ravel()
    return readback_coeffs


def score_energy_concentration(coeffs):
    """Score how concentrated/peaked the energy distribution is.

    Higher = more concentrated (fewer modes carry most energy).
    """
    energy = coeffs ** 2
    total = energy.sum()
    if total == 0:
        return 0.0
    # Normalized energy distribution
    p = energy / total
    # Gini coefficient (1 = all energy in one mode, 0 = uniform)
    sorted_p = np.sort(p)
    n = len(sorted_p)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_p) / (n * np.sum(sorted_p))) - (n + 1) / n
    return gini


def score_peak_sharpness(coeffs):
    """Score how sharp the peaks are in the coefficient spectrum."""
    abs_coeffs = np.abs(coeffs)
    # Ratio of max to mean
    mean_val = abs_coeffs.mean()
    if mean_val == 0:
        return 0.0
    return abs_coeffs.max() / mean_val


def score_sign_stability(coeffs):
    """Score based on how far coefficients are from zero (more decisive = more stable)."""
    return float(np.mean(np.abs(coeffs)))


def main():
    print()
    print("=" * 70)
    print("  PHASE 5: RESONANCE-TUNED ORACLE")
    print("  The 'Missing Step' Experiment: Harmonic Processing of EC Remainders")
    print(f"  {len(FREQUENCIES)} frequencies x 256 bits x 3 scoring methods")
    print("=" * 70)

    target_key = KeyInput(TARGET_HEX)
    actual_bits = np.array(target_key.as_bits, dtype=np.int8)

    # Compute public key
    K = G * target_key.as_int
    Kx = K.x()

    # Precompute bit bases
    print("\n  Precomputing EC bit bases...")
    bit_bases = [None] * 256
    bit_bases[0] = G
    for i in range(1, 256):
        bit_bases[i] = bit_bases[i - 1].double()

    # Pre-warm QR basis cache
    print(f"  Pre-building QR basis ({GRID_SIZE}x{GRID_SIZE})...")
    build_qr_sh_basis(GRID_SIZE, N_MODES)

    # ================================================================
    # SCORE EACH FREQUENCY
    # ================================================================
    csv_rows = []

    for freq in FREQUENCIES:
        print(f"\n  Frequency: {freq} MHz")
        t0 = time.time()

        correct_energy = 0
        correct_sharpness = 0
        correct_stability = 0

        for bit_idx in range(256):
            power = 255 - bit_idx
            P_i = bit_bases[power]
            neg_P_i = Point(CURVE, P_i.x(), (-P_i.y()) % P_FIELD)
            R = K + neg_P_i  # remainder if bit=1

            # Harmonic processing of each candidate
            # bit=0: use K.x directly
            coeffs_0 = key_to_resonated_coeffs(Kx, freq)
            # bit=1: use remainder.x
            if R == INFINITY:
                coeffs_1 = np.zeros(N_MODES)
            else:
                coeffs_1 = key_to_resonated_coeffs(R.x(), freq)

            # Score: energy concentration
            e0 = score_energy_concentration(coeffs_0)
            e1 = score_energy_concentration(coeffs_1)
            pred_energy = 1 if e1 > e0 else 0
            if pred_energy == actual_bits[bit_idx]:
                correct_energy += 1

            # Score: peak sharpness
            s0 = score_peak_sharpness(coeffs_0)
            s1 = score_peak_sharpness(coeffs_1)
            pred_sharp = 1 if s1 > s0 else 0
            if pred_sharp == actual_bits[bit_idx]:
                correct_sharpness += 1

            # Score: sign stability
            st0 = score_sign_stability(coeffs_0)
            st1 = score_sign_stability(coeffs_1)
            pred_stable = 1 if st1 > st0 else 0
            if pred_stable == actual_bits[bit_idx]:
                correct_stability += 1

        elapsed = time.time() - t0

        # Ensemble: majority of 3 scoring methods
        best = max(correct_energy, correct_sharpness, correct_stability)

        print(f"    Energy concentration: {correct_energy}/256 ({correct_energy-128:+d})")
        print(f"    Peak sharpness:       {correct_sharpness}/256 ({correct_sharpness-128:+d})")
        print(f"    Sign stability:       {correct_stability}/256 ({correct_stability-128:+d})")
        print(f"    Best method:          {best}/256 ({best-128:+d})")
        print(f"    Time: {elapsed:.1f}s")

        csv_rows.append({
            "frequency_mhz": freq,
            "energy_concentration": correct_energy,
            "peak_sharpness": correct_sharpness,
            "sign_stability": correct_stability,
            "best_method": best,
            "time_seconds": round(elapsed, 1),
        })

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY: RESONANCE FREQUENCY vs ACCURACY")
    print("=" * 70)

    print(f"\n  {'Freq (MHz)':>10s}  {'Energy':>8s}  {'Sharpness':>10s}  {'Stability':>10s}  {'Best':>6s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*6}")

    best_overall = 0
    best_freq = 0
    for row in csv_rows:
        f = row["frequency_mhz"]
        e = row["energy_concentration"]
        s = row["peak_sharpness"]
        st = row["sign_stability"]
        b = row["best_method"]
        marker = " <--" if b > 140 else ""
        print(f"  {f:10d}  {e:5d}/256  {s:7d}/256  {st:7d}/256  {b:3d}/256{marker}")
        if b > best_overall:
            best_overall = b
            best_freq = f

    print(f"\n  Best frequency: {best_freq} MHz with {best_overall}/256 ({best_overall-128:+d})")

    if best_overall > 145:
        print(f"\n  *** SIGNAL AT {best_freq} MHz -- VALIDATE ON MULTIPLE KEYS ***")
    elif best_overall > 135:
        print(f"\n  Marginal signal at {best_freq} MHz -- might be noise, worth investigating")
    else:
        print(f"\n  No frequency shows signal above noise floor.")
        print(f"  Harmonic processing does not amplify EC remainder differences.")

    # ================================================================
    # WRITE CSV
    # ================================================================
    csv_path = "/Users/kjm/Desktop/phase5_resonance_oracle.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  Results written to {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
