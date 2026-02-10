"""Phase 1: Multi-Key Oracle Validation (THE GATEKEEPER).

The x_mod_3 oracle scored 149/256 (+21) on ONE key.
Is that signal or noise?

Test: run all 256 oracles across 10 different random keys.
If x_mod_3 (or ANY oracle) consistently scores above random,
that's real signal. If it drops to ~128, it was variance.

Statistical test: one-sample t-test per oracle, Bonferroni corrected.
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
from superposition_256_oracles import (
    build_oracle_index,
    evaluate_all_256_oracles,
)

G = SECP256k1.generator
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()

NUM_KEYS = 10
NUM_ORACLES = 256
NUM_BITS = 256
BONFERRONI_ALPHA = 0.05 / NUM_ORACLES  # 0.000195


def main():
    print()
    print("=" * 70)
    print("  PHASE 1: MULTI-KEY ORACLE VALIDATION")
    print(f"  {NUM_ORACLES} oracles x {NUM_KEYS} keys x {NUM_BITS} bits")
    print(f"  Bonferroni threshold: p < {BONFERRONI_ALPHA:.6f}")
    print("=" * 70)

    names = build_oracle_index()

    # Precompute bit bases (same for all keys)
    print("\n  Precomputing 256 bit-basis EC points...")
    bit_bases = [None] * 256
    bit_bases[0] = G
    for i in range(1, 256):
        bit_bases[i] = bit_bases[i - 1].double()

    # Results: (NUM_ORACLES, NUM_KEYS) matrix of accuracies
    all_accuracies = np.zeros((NUM_ORACLES, NUM_KEYS), dtype=np.float64)

    for ki in range(NUM_KEYS):
        # Generate random key
        key = KeyInput(secrets.token_bytes(32))
        actual_bits = np.array(key.as_bits, dtype=np.int8)

        # Compute public key
        K = G * key.as_int
        Kx, Ky = K.x(), K.y()

        print(f"\n  Key {ki+1}/{NUM_KEYS}: {key.as_hex[:16]}...")
        t0 = time.time()

        # Score base point K (for bit=0 at every position)
        base_scores = evaluate_all_256_oracles(Kx, Ky)

        # For each bit position, compute remainder and score
        oracle_correct = np.zeros(NUM_ORACLES, dtype=np.int32)

        for bit_idx in range(NUM_BITS):
            power = 255 - bit_idx
            P_i = bit_bases[power]
            neg_P_i = Point(CURVE, P_i.x(), (-P_i.y()) % P_FIELD)
            R = K + neg_P_i

            if R == INFINITY:
                r_scores = np.zeros(NUM_ORACLES)
            else:
                r_scores = evaluate_all_256_oracles(R.x(), R.y())

            # Each oracle predicts bit=1 if score_1 > score_0
            for oi in range(NUM_ORACLES):
                predicted = 1 if r_scores[oi] > base_scores[oi] else 0
                if predicted == actual_bits[bit_idx]:
                    oracle_correct[oi] += 1

        all_accuracies[:, ki] = oracle_correct
        elapsed = time.time() - t0
        best_oi = np.argmax(oracle_correct)
        print(f"    Done in {elapsed:.1f}s. Best oracle: {names[best_oi]} = {oracle_correct[best_oi]}/256")

    # ================================================================
    # STATISTICAL ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  STATISTICAL ANALYSIS")
    print("=" * 70)

    means = np.mean(all_accuracies, axis=1)
    stds = np.std(all_accuracies, axis=1, ddof=1)

    # One-sample t-test per oracle: is mean significantly > 128?
    t_stats = np.zeros(NUM_ORACLES)
    p_values = np.ones(NUM_ORACLES)
    for oi in range(NUM_ORACLES):
        if stds[oi] > 0:
            t_stat, p_val = stats.ttest_1samp(all_accuracies[oi], 128)
            t_stats[oi] = t_stat
            # One-sided: we only care about above random
            p_values[oi] = p_val / 2 if t_stat > 0 else 1.0
        else:
            t_stats[oi] = 0
            p_values[oi] = 1.0

    # Sort by mean accuracy
    sorted_idx = np.argsort(means)[::-1]

    print(f"\n  Top 20 oracles (mean across {NUM_KEYS} keys):")
    print(f"  {'Rank':>4s}  {'Oracle':40s}  {'Mean':>6s}  {'Std':>5s}  {'t':>6s}  {'p':>10s}  {'Sig?':>5s}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*5}")
    for rank, oi in enumerate(sorted_idx[:20]):
        sig = "***" if p_values[oi] < BONFERRONI_ALPHA else ""
        print(f"  {rank+1:4d}  {names[oi]:40s}  {means[oi]:5.1f}  {stds[oi]:5.1f}  {t_stats[oi]:6.2f}  {p_values[oi]:10.6f}  {sig:>5s}")

    print(f"\n  Bottom 5:")
    for rank, oi in enumerate(sorted_idx[-5:]):
        print(f"  {NUM_ORACLES-4+rank:4d}  {names[oi]:40s}  {means[oi]:5.1f}  {stds[oi]:5.1f}")

    # Track x_mod_3 specifically
    xmod3_idx = None
    for oi, name in enumerate(names):
        if "x_mod_3" in name:
            xmod3_idx = oi
            break

    if xmod3_idx is not None:
        print(f"\n  x_mod_3 TRACKER:")
        print(f"    Per-key scores: {all_accuracies[xmod3_idx].astype(int)}")
        print(f"    Mean: {means[xmod3_idx]:.1f}/256 (+{means[xmod3_idx]-128:.1f})")
        print(f"    Std:  {stds[xmod3_idx]:.1f}")
        print(f"    t-stat: {t_stats[xmod3_idx]:.3f}")
        print(f"    p-value: {p_values[xmod3_idx]:.6f}")
        if p_values[xmod3_idx] < BONFERRONI_ALPHA:
            print(f"    VERDICT: REAL SIGNAL (survives Bonferroni)")
        else:
            print(f"    VERDICT: NOISE (p > {BONFERRONI_ALPHA:.6f})")

    # Count significant oracles
    n_significant = np.sum(p_values < BONFERRONI_ALPHA)
    n_marginal = np.sum(p_values < 0.05)

    print(f"\n  SUMMARY:")
    print(f"  Oracles surviving Bonferroni (p < {BONFERRONI_ALPHA:.6f}): {n_significant}/256")
    print(f"  Oracles at nominal 5% (p < 0.05): {n_marginal}/256 (expected by chance: ~13)")
    print(f"  Overall mean accuracy: {np.mean(means):.1f}/256")
    print(f"  Overall std of means: {np.std(means):.2f}")

    if n_significant > 0:
        print(f"\n  *** SIGNAL DETECTED: {n_significant} oracle(s) survive Bonferroni ***")
        for oi in range(NUM_ORACLES):
            if p_values[oi] < BONFERRONI_ALPHA:
                print(f"      {names[oi]}: mean={means[oi]:.1f}, p={p_values[oi]:.8f}")
    else:
        print(f"\n  NO SIGNAL: Zero oracles survive multi-key validation.")
        print(f"  The x_mod_3 result (+21 on one key) was statistical noise.")

    # ================================================================
    # WRITE CSV
    # ================================================================
    csv_path = "/Users/kjm/Desktop/phase1_multikey_oracle.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["oracle_name"] + [f"key{i+1}" for i in range(NUM_KEYS)] + [
            "mean", "std", "t_stat", "p_value", "significant"
        ]
        writer.writerow(header)
        for oi in sorted_idx:
            row = [names[oi]] + [int(all_accuracies[oi, ki]) for ki in range(NUM_KEYS)]
            row += [f"{means[oi]:.2f}", f"{stds[oi]:.2f}",
                    f"{t_stats[oi]:.4f}", f"{p_values[oi]:.8f}",
                    "YES" if p_values[oi] < BONFERRONI_ALPHA else "no"]
            writer.writerow(row)

    print(f"\n  Results written to {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
