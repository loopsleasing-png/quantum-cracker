"""Phase 3: EC Double-and-Add Trail Analysis.

EC multiplication processes bits sequentially via double-and-add.
The intermediate points might leak bit values through their
mathematical properties. This is software side-channel analysis.

For each key:
  1. Simulate double-and-add with instrumentation
  2. Capture all intermediate EC points
  3. Score each point with oracles
  4. Compare "double-only" vs "double+add" distributions
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

G = SECP256k1.generator
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()
ORDER = SECP256k1.order

FIRST_16_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
NUM_KEYS = 10


def ec_multiply_with_trail(k):
    """Double-and-add with full trail capture.

    Returns: (result_point, trail) where trail is list of
             (operation, step_idx, x, y, bit_value)
    """
    bits = bin(k)[2:]  # MSB first, no leading zeros
    trail = []

    # First bit is always 1 (k > 0)
    R = G
    trail.append(("init", 0, R.x(), R.y(), 1))

    for i in range(1, len(bits)):
        # Always double
        R = R.double()
        trail.append(("double", i, R.x(), R.y(), int(bits[i])))

        if bits[i] == '1':
            # Add G
            R = R + G
            trail.append(("add", i, R.x(), R.y(), 1))

    return R, trail


def score_point(x, y):
    """Simplified oracle scoring for trail analysis."""
    scores = {}

    # Hamming weight of x
    scores["x_hw"] = bin(x).count("1")
    scores["y_hw"] = bin(y).count("1")

    # x mod small primes
    for p in FIRST_16_PRIMES[:8]:
        scores[f"x_mod_{p}"] = x % p

    # Trailing zeros
    if x == 0:
        scores["x_trailing"] = 256
    else:
        scores["x_trailing"] = (x & -x).bit_length() - 1

    # Nibble entropy of x
    x_hex = f"{x:064x}"
    nibble_counts = np.array([x_hex.count(c) for c in "0123456789abcdef"])
    freq = nibble_counts / 64.0
    entropy = -np.sum(freq[freq > 0] * np.log2(freq[freq > 0]))
    scores["x_entropy"] = entropy

    # XOR of x and y hamming weight
    scores["xor_hw"] = bin(x ^ y).count("1")

    # DFT first component
    x_bits = np.array([(x >> (255 - i)) & 1 for i in range(256)], dtype=float) * 2 - 1
    fft_mag = np.abs(np.fft.rfft(x_bits))
    scores["x_dft_1"] = fft_mag[1] if len(fft_mag) > 1 else 0

    return scores


def main():
    print()
    print("=" * 70)
    print("  PHASE 3: EC DOUBLE-AND-ADD TRAIL ANALYSIS")
    print(f"  {NUM_KEYS} keys, full trail capture + oracle scoring")
    print("=" * 70)

    all_double_scores = {k: [] for k in ["x_hw", "y_hw", "x_trailing", "x_entropy",
                                           "xor_hw", "x_dft_1"] + [f"x_mod_{p}" for p in FIRST_16_PRIMES[:8]]}
    all_add_scores = {k: [] for k in all_double_scores}

    csv_rows = []

    for ki in range(NUM_KEYS):
        key = KeyInput(secrets.token_bytes(32))
        k_int = key.as_int
        if k_int == 0:
            continue

        print(f"\n  Key {ki+1}/{NUM_KEYS}: {key.as_hex[:16]}...")
        t0 = time.time()

        result, trail = ec_multiply_with_trail(k_int)

        # Verify correctness
        expected = G * k_int
        assert result.x() == expected.x(), "Trail multiply gave wrong result!"

        n_doubles = sum(1 for op, *_ in trail if op == "double")
        n_adds = sum(1 for op, *_ in trail if op == "add")
        print(f"    Trail: {len(trail)} steps ({n_doubles} doubles, {n_adds} adds)")

        # Score each intermediate point
        for op, step_idx, x, y, bit_val in trail:
            if op == "init":
                continue

            scores = score_point(x, y)

            # Categorize
            target_dict = all_add_scores if op == "add" else all_double_scores
            for k, v in scores.items():
                target_dict[k].append(v)

            csv_rows.append({
                "key_index": ki,
                "step_index": step_idx,
                "operation": op,
                "bit_value": bit_val,
                **scores
            })

        elapsed = time.time() - t0
        print(f"    Scored in {elapsed:.1f}s")

    # ================================================================
    # STATISTICAL COMPARISON: double vs add
    # ================================================================
    print("\n" + "=" * 70)
    print("  DOUBLE vs ADD: INTERMEDIATE POINT COMPARISON")
    print("=" * 70)

    print(f"\n  {'Metric':18s}  {'Double Mean':>12s}  {'Add Mean':>10s}  {'t-stat':>8s}  {'p-value':>10s}  {'Sig?':>5s}")
    print(f"  {'-'*18}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*5}")

    significant_metrics = []
    for metric in all_double_scores:
        d_vals = np.array(all_double_scores[metric], dtype=float)
        a_vals = np.array(all_add_scores[metric], dtype=float)

        if len(d_vals) == 0 or len(a_vals) == 0:
            continue

        t_stat, p_val = stats.ttest_ind(d_vals, a_vals)
        sig = "***" if p_val < 0.001 else ("*" if p_val < 0.05 else "")

        print(f"  {metric:18s}  {np.mean(d_vals):12.4f}  {np.mean(a_vals):10.4f}  "
              f"{t_stat:8.3f}  {p_val:10.6f}  {sig:>5s}")

        if p_val < 0.001:
            significant_metrics.append((metric, t_stat, p_val))

    # ================================================================
    # BIT-VALUE ANALYSIS: does the intermediate point after "add"
    # differ depending on whether the NEXT bit is 0 or 1?
    # ================================================================
    print("\n" + "=" * 70)
    print("  PREDICTIVE POWER: Can trail points predict NEXT bit?")
    print("=" * 70)

    # For "double" steps: does the point predict the current bit?
    # (bit_val in trail tells us which bit this step processes)
    double_bit0 = {k: [] for k in all_double_scores}
    double_bit1 = {k: [] for k in all_double_scores}

    for row in csv_rows:
        if row["operation"] == "double":
            target = double_bit1 if row["bit_value"] == 1 else double_bit0
            for k in all_double_scores:
                target[k].append(row[k])

    print(f"\n  Double-step points: bit=0 ({len(double_bit0['x_hw'])}) vs bit=1 ({len(double_bit1['x_hw'])})")
    print(f"\n  {'Metric':18s}  {'bit=0 Mean':>11s}  {'bit=1 Mean':>11s}  {'t-stat':>8s}  {'p-value':>10s}  {'Sig?':>5s}")
    print(f"  {'-'*18}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*10}  {'-'*5}")

    for metric in all_double_scores:
        b0 = np.array(double_bit0[metric], dtype=float)
        b1 = np.array(double_bit1[metric], dtype=float)
        if len(b0) == 0 or len(b1) == 0:
            continue
        t_stat, p_val = stats.ttest_ind(b0, b1)
        sig = "***" if p_val < 0.001 else ("*" if p_val < 0.05 else "")
        print(f"  {metric:18s}  {np.mean(b0):11.4f}  {np.mean(b1):11.4f}  "
              f"{t_stat:8.3f}  {p_val:10.6f}  {sig:>5s}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if significant_metrics:
        print(f"\n  {len(significant_metrics)} metric(s) show significant double vs add difference:")
        for m, t, p in significant_metrics:
            print(f"    {m}: t={t:.3f}, p={p:.6f}")
        print(f"\n  NOTE: This means the intermediate points have different properties")
        print(f"  depending on the operation. This IS a side-channel in the math.")
    else:
        print(f"\n  No significant differences between double and add intermediate points.")
        print(f"  The EC group operation produces statistically identical points")
        print(f"  regardless of the operation type.")

    # ================================================================
    # WRITE CSV
    # ================================================================
    csv_path = "/Users/kjm/Desktop/phase3_trail_analysis.csv"
    with open(csv_path, "w", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n  Results written to {csv_path} ({len(csv_rows)} rows)")
    print("=" * 70)


if __name__ == "__main__":
    main()
