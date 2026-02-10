"""Phase 4: Per-Bit Cross-Method Consistency.

Meta-analysis: collect per-bit predictions from ALL prior methods.
If specific bits are ALWAYS predicted correctly across all methods,
those bits have exploitable structure in EC multiplication.

Under null (all random): P(all N methods correct) = 1/2^N per bit.
With N=5 methods: expected 256/32 = 8 bits "all correct."
Significantly more = shared signal.
"""

import csv
import secrets
import sys
import time

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, "src")

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import INFINITY, Point

from quantum_cracker.core.key_interface import KeyInput
from superposition_256_oracles import (
    build_oracle_index,
    evaluate_all_256_oracles,
)

G = SECP256k1.generator
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()

TARGET_HEX = "06d88f2148757a251dd0ea0e6c4584e159a60cfd3f7217c7b0b111adec0efbca"


def method_256_oracle_weighted(all_scores, actual_bits):
    """Weighted vote across 256 oracles."""
    oracle_acc = np.zeros(256)
    oracle_preds = np.zeros((256, 256), dtype=np.int8)

    for oi in range(256):
        pred = np.where(all_scores[oi, :, 1] > all_scores[oi, :, 0], 1, 0).astype(np.int8)
        oracle_preds[oi] = pred
        oracle_acc[oi] = np.sum(pred == actual_bits)

    weights = oracle_acc - 128
    weighted_sum = np.zeros(256)
    for oi in range(256):
        if weights[oi] > 0:
            weighted_sum += weights[oi] * (oracle_preds[oi] * 2 - 1)

    return np.where(weighted_sum > 0, 1, 0).astype(np.int8)


def method_top10_vote(all_scores, actual_bits):
    """Top-10 oracles by accuracy, majority vote."""
    oracle_acc = np.zeros(256)
    oracle_preds = np.zeros((256, 256), dtype=np.int8)

    for oi in range(256):
        pred = np.where(all_scores[oi, :, 1] > all_scores[oi, :, 0], 1, 0).astype(np.int8)
        oracle_preds[oi] = pred
        oracle_acc[oi] = np.sum(pred == actual_bits)

    top10 = np.argsort(oracle_acc)[-10:]
    vote = oracle_preds[top10].sum(axis=0)
    return np.where(vote > 5, 1, 0).astype(np.int8)


def method_majority_all(all_scores, actual_bits):
    """Majority vote across all 256 oracles."""
    oracle_preds = np.zeros((256, 256), dtype=np.int8)
    for oi in range(256):
        oracle_preds[oi] = np.where(all_scores[oi, :, 1] > all_scores[oi, :, 0], 1, 0).astype(np.int8)
    vote = oracle_preds.sum(axis=0)
    return np.where(vote > 128, 1, 0).astype(np.int8)


def method_single_best(all_scores, actual_bits):
    """Single best oracle prediction."""
    oracle_acc = np.zeros(256)
    oracle_preds = np.zeros((256, 256), dtype=np.int8)
    for oi in range(256):
        pred = np.where(all_scores[oi, :, 1] > all_scores[oi, :, 0], 1, 0).astype(np.int8)
        oracle_preds[oi] = pred
        oracle_acc[oi] = np.sum(pred == actual_bits)
    best = np.argmax(oracle_acc)
    return oracle_preds[best]


def method_random(n_bits=256):
    """Random baseline."""
    return np.random.randint(0, 2, size=n_bits).astype(np.int8)


def main():
    print()
    print("=" * 70)
    print("  PHASE 4: PER-BIT CROSS-METHOD CONSISTENCY")
    print("=" * 70)

    target_key = KeyInput(TARGET_HEX)
    actual_bits = np.array(target_key.as_bits, dtype=np.int8)

    # Compute public key
    K = G * target_key.as_int
    Kx, Ky = K.x(), K.y()

    # Precompute bit bases
    print("\n  Precomputing EC bit bases...")
    bit_bases = [None] * 256
    bit_bases[0] = G
    for i in range(1, 256):
        bit_bases[i] = bit_bases[i - 1].double()

    # Run 256-oracle scoring (same as Phase 1 but single key)
    print("  Running 256-oracle scoring...")
    t0 = time.time()

    base_scores = evaluate_all_256_oracles(Kx, Ky)
    all_scores = np.zeros((256, 256, 2), dtype=np.float64)
    for bit_idx in range(256):
        all_scores[:, bit_idx, 0] = base_scores

    for bit_idx in range(256):
        power = 255 - bit_idx
        P_i = bit_bases[power]
        neg_P_i = Point(CURVE, P_i.x(), (-P_i.y()) % P_FIELD)
        R = K + neg_P_i
        if R == INFINITY:
            all_scores[:, bit_idx, 1] = 0
        else:
            all_scores[:, bit_idx, 1] = evaluate_all_256_oracles(R.x(), R.y())

    print(f"  Oracle scoring done in {time.time()-t0:.1f}s")

    # ================================================================
    # COLLECT PREDICTIONS FROM ALL METHODS
    # ================================================================
    print("\n  Generating predictions from 8 methods...")

    np.random.seed(42)
    methods = {
        "weighted_vote": method_256_oracle_weighted(all_scores, actual_bits),
        "top10_vote": method_top10_vote(all_scores, actual_bits),
        "majority_all": method_majority_all(all_scores, actual_bits),
        "single_best": method_single_best(all_scores, actual_bits),
        "random_1": method_random(),
        "random_2": method_random(),
        "random_3": method_random(),
        "random_4": method_random(),
    }

    method_names = list(methods.keys())
    n_methods = len(method_names)
    n_real_methods = 4  # first 4 are real, last 4 are random baselines

    # Per-method accuracy
    print(f"\n  Per-method accuracy:")
    for name, pred in methods.items():
        correct = np.sum(pred == actual_bits)
        print(f"    {name:20s}: {correct}/256 ({correct/256*100:.1f}%)")

    # ================================================================
    # PER-BIT ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  PER-BIT CONSISTENCY ANALYSIS")
    print("=" * 70)

    # Build prediction matrix: (n_methods, 256)
    pred_matrix = np.array([methods[name] for name in method_names], dtype=np.int8)
    correct_matrix = (pred_matrix == actual_bits[np.newaxis, :]).astype(np.int8)

    # For real methods only (first 4)
    real_correct = correct_matrix[:n_real_methods]
    all_correct_count = real_correct.sum(axis=0)  # how many real methods got each bit right

    # Bits correct by ALL real methods
    all_right = np.sum(all_correct_count == n_real_methods)
    expected_all_right = 256 * (0.5 ** n_real_methods)  # = 16 for 4 methods

    # Bits correct by NO real method
    none_right = np.sum(all_correct_count == 0)
    expected_none_right = expected_all_right  # also 16

    print(f"\n  {n_real_methods} real methods x 256 bits:")
    print(f"  Bits correct by ALL {n_real_methods} methods: {all_right} (expected under random: {expected_all_right:.1f})")
    print(f"  Bits correct by NO method:       {none_right} (expected: {expected_all_right:.1f})")

    # Distribution
    for count in range(n_real_methods + 1):
        n_bits = np.sum(all_correct_count == count)
        expected = 256 * (0.5 ** n_real_methods) * np.math.comb(n_real_methods, count) if hasattr(np, 'math') else 256 * (0.5 ** n_real_methods)
        # Binomial expected
        from math import comb
        expected = 256 * (0.5 ** n_real_methods) * comb(n_real_methods, count)
        print(f"    Correct by {count}/{n_real_methods}: {n_bits:3d} bits (expected: {expected:.1f})")

    # Pairwise agreement between methods
    print(f"\n  Pairwise agreement (fraction of bits where both methods predict same):")
    for i in range(n_real_methods):
        for j in range(i + 1, n_real_methods):
            agree = np.sum(pred_matrix[i] == pred_matrix[j]) / 256
            print(f"    {method_names[i]:20s} vs {method_names[j]:20s}: {agree:.3f}")

    # Chi-squared test: is the distribution of all_correct_count
    # significantly different from binomial(n_real_methods, 0.5)?
    from math import comb
    observed = np.array([np.sum(all_correct_count == c) for c in range(n_real_methods + 1)])
    expected = np.array([256 * (0.5 ** n_real_methods) * comb(n_real_methods, c)
                         for c in range(n_real_methods + 1)])
    chi2, p_val = scipy_stats.chisquare(observed, f_exp=expected)
    print(f"\n  Chi-squared test (observed vs binomial expected):")
    print(f"    chi2 = {chi2:.3f}, p = {p_val:.6f}")
    if p_val < 0.05:
        print(f"    *** Distribution differs from random -- methods share structure ***")
    else:
        print(f"    Distribution consistent with random (methods are independent)")

    # ================================================================
    # IDENTIFY ALWAYS-CORRECT BITS
    # ================================================================
    always_correct = np.where(all_correct_count == n_real_methods)[0]
    if len(always_correct) > expected_all_right * 2:
        print(f"\n  NOTABLE: {len(always_correct)} bits correct by all methods (expected ~{expected_all_right:.0f})")
        print(f"  These bits: {always_correct.tolist()[:20]}{'...' if len(always_correct) > 20 else ''}")
    else:
        print(f"\n  Always-correct bits: {len(always_correct)} (within expected range)")

    # ================================================================
    # WRITE CSV
    # ================================================================
    csv_path = "/Users/kjm/Desktop/phase4_consistency.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["bit_position", "actual_bit"] + [f"{name}_correct" for name in method_names] + ["total_real_correct"]
        writer.writerow(header)
        for b in range(256):
            row = [b, int(actual_bits[b])]
            row += [int(correct_matrix[mi, b]) for mi in range(n_methods)]
            row += [int(all_correct_count[b])]
            writer.writerow(row)

    print(f"\n  Results written to {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
