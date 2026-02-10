"""Phase 5 Multi-Key Validation.

The 50 MHz peak sharpness scored 143/256 (+15) on one key.
Is that real or noise? Run on 20 random keys to find out.

Bonferroni threshold for 27 tests (9 freq x 3 methods): p < 0.05/27 = 0.00185
For peak sharpness at 50 MHz alone, we need t-test vs 128.
"""

import csv
import secrets
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, "src")

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import INFINITY, Point

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.utils.math_helpers import build_qr_sh_basis

G = SECP256k1.generator
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()
ORDER = SECP256k1.order

GRID_SIZE = 20
N_MODES = 256
RESONANCE_STEPS = 10
RESONANCE_STRENGTH = 0.05
NUM_KEYS = 20
FREQUENCIES = [10, 50, 100, 156]  # focused on promising range


def apply_resonance(grid_2d, freq, steps=RESONANCE_STEPS, strength=RESONANCE_STRENGTH):
    gs = grid_2d.shape[0]
    theta = np.linspace(0, np.pi, gs)
    phi = np.linspace(0, 2 * np.pi, gs)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    result = grid_2d.copy()
    for t in range(steps):
        vibration = np.sin(freq * phi_grid + t * 0.1) * np.cos(freq * theta_grid)
        result *= (1.0 + vibration * strength)
    return result


def key_to_resonated_coeffs(key_int, freq, basis):
    key_hex = f"{key_int % (2**256):064x}"
    key = KeyInput(key_hex)
    bits = np.array(key.as_bits, dtype=np.float64)
    coeffs = 2.0 * bits - 1.0
    angular_field = (basis @ coeffs).reshape(GRID_SIZE, GRID_SIZE)
    max_val = np.abs(angular_field).max()
    if max_val > 0:
        angular_field /= max_val
    resonated = apply_resonance(angular_field, freq)
    return basis.T @ resonated.ravel()


def score_peak_sharpness(coeffs):
    abs_coeffs = np.abs(coeffs)
    mean_val = abs_coeffs.mean()
    if mean_val == 0:
        return 0.0
    return abs_coeffs.max() / mean_val


def score_energy_concentration(coeffs):
    energy = coeffs ** 2
    total = energy.sum()
    if total == 0:
        return 0.0
    p = energy / total
    sorted_p = np.sort(p)
    n = len(sorted_p)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_p) / (n * np.sum(sorted_p))) - (n + 1) / n


def main():
    print()
    print("=" * 70)
    print("  PHASE 5 MULTI-KEY VALIDATION")
    print(f"  {NUM_KEYS} random keys x {len(FREQUENCIES)} frequencies")
    print("=" * 70)

    basis = build_qr_sh_basis(GRID_SIZE, N_MODES)

    # Precompute bit bases
    bit_bases = [None] * 256
    bit_bases[0] = G
    for i in range(1, 256):
        bit_bases[i] = bit_bases[i - 1].double()

    # Results: (NUM_KEYS, len(FREQUENCIES), 3) -- 3 scoring methods
    results = np.zeros((NUM_KEYS, len(FREQUENCIES), 3), dtype=int)
    method_names = ["energy", "sharpness", "stability"]

    t_total = time.time()

    for ki in range(NUM_KEYS):
        key_bytes = secrets.token_bytes(32)
        key = KeyInput(key_bytes)
        actual_bits = np.array(key.as_bits, dtype=np.int8)
        K = G * key.as_int
        Kx = K.x()

        print(f"\n  Key {ki+1}/{NUM_KEYS}: {key.as_hex[:16]}...")

        for fi, freq in enumerate(FREQUENCIES):
            t0 = time.time()
            correct_energy = 0
            correct_sharp = 0
            correct_stability = 0

            for bit_idx in range(256):
                power = 255 - bit_idx
                P_i = bit_bases[power]
                neg_P_i = Point(CURVE, P_i.x(), (-P_i.y()) % P_FIELD)
                R = K + neg_P_i

                coeffs_0 = key_to_resonated_coeffs(Kx, freq, basis)
                if R == INFINITY:
                    coeffs_1 = np.zeros(N_MODES)
                else:
                    coeffs_1 = key_to_resonated_coeffs(R.x(), freq, basis)

                e0 = score_energy_concentration(coeffs_0)
                e1 = score_energy_concentration(coeffs_1)
                if (1 if e1 > e0 else 0) == actual_bits[bit_idx]:
                    correct_energy += 1

                s0 = score_peak_sharpness(coeffs_0)
                s1 = score_peak_sharpness(coeffs_1)
                if (1 if s1 > s0 else 0) == actual_bits[bit_idx]:
                    correct_sharp += 1

                st0 = float(np.mean(np.abs(coeffs_0)))
                st1 = float(np.mean(np.abs(coeffs_1)))
                if (1 if st1 > st0 else 0) == actual_bits[bit_idx]:
                    correct_stability += 1

            results[ki, fi] = [correct_energy, correct_sharp, correct_stability]
            dt = time.time() - t0
            print(f"    {freq:4d} MHz: energy={correct_energy}, sharp={correct_sharp}, "
                  f"stable={correct_stability} ({dt:.1f}s)")

    total_time = time.time() - t_total
    print(f"\n  Total time: {total_time:.1f}s")

    # ================================================================
    # STATISTICAL ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  STATISTICAL ANALYSIS")
    print("=" * 70)

    print(f"\n  {'Freq':>6s}  {'Method':>12s}  {'Mean':>6s}  {'Std':>5s}  {'t-stat':>7s}  {'p-value':>10s}  {'Sig?':>5s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*5}")

    csv_rows = []
    for fi, freq in enumerate(FREQUENCIES):
        for mi, mname in enumerate(method_names):
            scores = results[:, fi, mi]
            mean_val = scores.mean()
            std_val = scores.std(ddof=1)
            if std_val > 0:
                t_stat, p_val = stats.ttest_1samp(scores, 128)
                p_val = p_val / 2 if t_stat > 0 else 1.0
            else:
                t_stat = 0
                p_val = 1.0

            sig = "***" if p_val < 0.001 else ("*" if p_val < 0.05 else "")
            print(f"  {freq:6d}  {mname:>12s}  {mean_val:5.1f}  {std_val:5.1f}  {t_stat:7.3f}  {p_val:10.6f}  {sig:>5s}")

            csv_rows.append({
                "frequency": freq,
                "method": mname,
                "mean": f"{mean_val:.2f}",
                "std": f"{std_val:.2f}",
                "t_stat": f"{t_stat:.4f}",
                "p_value": f"{p_val:.8f}",
                "per_key_scores": ",".join(str(int(s)) for s in scores),
            })

    # Best result
    best_mean = 0
    best_desc = ""
    for fi, freq in enumerate(FREQUENCIES):
        for mi, mname in enumerate(method_names):
            m = results[:, fi, mi].mean()
            if m > best_mean:
                best_mean = m
                best_desc = f"{freq} MHz / {mname}"

    print(f"\n  Best overall: {best_desc} = {best_mean:.1f}/256 ({best_mean-128:+.1f})")

    if best_mean > 135:
        print(f"  *** SIGNAL PERSISTS across {NUM_KEYS} keys ***")
    else:
        print(f"  No signal. The 143/256 was noise.")

    # Write CSV
    csv_path = "/Users/kjm/Desktop/phase5_multikey.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n  Written to {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
