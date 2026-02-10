"""Timing Side-Channel Analysis of EC Scalar Multiplication.

Does python-ecdsa's scalar multiplication leak timing information
based on the private key's bit pattern?

If EC multiply takes different time for different keys, an attacker
measuring timing could recover key bits. This is a REAL attack
vector (Kocher 1996, Brumley & Tuveri 2011).

We measure:
1. Time for ec_multiply with keys of varying Hamming weight
2. Time for keys with specific bit patterns (runs of 0s/1s)
3. Time for keys near powers of 2
4. Statistical analysis: can timing predict any key property?

Modern implementations use constant-time algorithms. Does python-ecdsa?
"""

import secrets
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, "src")

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import Point

G = SECP256k1.generator
CURVE = SECP256k1.curve
ORDER = SECP256k1.order

N_SAMPLES = 200
N_WARMUP = 10


def time_multiply(k, n_reps=5):
    """Time a single EC scalar multiplication, averaging n_reps runs."""
    # Warmup
    _ = G * k

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        _ = G * k
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    return np.median(times)  # median is more robust than mean


def main():
    print()
    print("=" * 78)
    print("  TIMING SIDE-CHANNEL ANALYSIS")
    print("  Does EC multiply leak timing information about the key?")
    print("=" * 78)

    # Warmup the CPU
    print(f"\n  Warming up ({N_WARMUP} multiplies)...")
    for _ in range(N_WARMUP):
        _ = G * secrets.randbelow(ORDER)

    # ================================================================
    # TEST 1: Hamming Weight vs Timing
    # ================================================================
    print(f"\n  TEST 1: Hamming weight vs timing ({N_SAMPLES} samples)")

    hw_data = []
    time_data = []

    for _ in range(N_SAMPLES):
        k = secrets.randbelow(ORDER - 1) + 1
        hw = bin(k).count("1")
        t = time_multiply(k)
        hw_data.append(hw)
        time_data.append(t)

    hw_arr = np.array(hw_data)
    time_arr = np.array(time_data)

    # Pearson correlation
    r_hw, p_hw = stats.pearsonr(hw_arr, time_arr)
    print(f"  Hamming weight range: {hw_arr.min()} - {hw_arr.max()}")
    print(f"  Timing range: {time_arr.min()/1e6:.2f} - {time_arr.max()/1e6:.2f} ms")
    print(f"  Pearson correlation: r = {r_hw:.4f}, p = {p_hw:.4f}")
    if abs(r_hw) > 0.3 and p_hw < 0.01:
        print(f"  *** TIMING LEAK: Hamming weight correlates with timing ***")
    else:
        print(f"  No correlation detected.")

    # ================================================================
    # TEST 2: Bit length vs Timing
    # ================================================================
    print(f"\n  TEST 2: Bit length vs timing")

    bl_data = []
    bl_time = []

    for bits in range(200, 257):
        for _ in range(5):
            k = secrets.randbits(bits) % ORDER
            if k == 0: k = 1
            t = time_multiply(k)
            bl_data.append(bits)
            bl_time.append(t)

    bl_arr = np.array(bl_data)
    bl_t_arr = np.array(bl_time)

    r_bl, p_bl = stats.pearsonr(bl_arr, bl_t_arr)
    print(f"  Bit length range: {bl_arr.min()} - {bl_arr.max()}")
    print(f"  Pearson correlation: r = {r_bl:.4f}, p = {p_bl:.4f}")
    if abs(r_bl) > 0.3 and p_bl < 0.01:
        print(f"  *** TIMING LEAK: Bit length correlates with timing ***")
    else:
        print(f"  No correlation detected.")

    # ================================================================
    # TEST 3: Specific bit patterns
    # ================================================================
    print(f"\n  TEST 3: Specific bit patterns")

    # All 1s (max hamming weight) vs alternating (50% hw)
    patterns = {
        "all_1s": (1 << 256) - 1,
        "alternating_10": int("10" * 128, 2),
        "alternating_01": int("01" * 128, 2),
        "low_hw_8": (1 << 255) | (1 << 128) | (1 << 64) | (1 << 32) | (1 << 16) | (1 << 8) | (1 << 4) | 1,
        "power_of_2": 1 << 128,
        "near_order": ORDER - 1,
        "small_key": 42,
        "random_1": secrets.randbelow(ORDER),
        "random_2": secrets.randbelow(ORDER),
    }

    pattern_times = {}
    print(f"  {'Pattern':20s}  {'HW':>5s}  {'Time (ms)':>10s}  {'Std (ms)':>9s}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*10}  {'-'*9}")

    for name, k in patterns.items():
        k = k % ORDER
        if k == 0: k = 1
        times = []
        for _ in range(20):
            t = time_multiply(k, n_reps=3)
            times.append(t)
        mean_t = np.mean(times) / 1e6  # to ms
        std_t = np.std(times) / 1e6
        hw = bin(k).count("1")
        pattern_times[name] = times
        print(f"  {name:20s}  {hw:5d}  {mean_t:10.3f}  {std_t:9.3f}")

    # Compare all_1s vs small_key timing
    t_stat, p_val = stats.ttest_ind(pattern_times["all_1s"], pattern_times["small_key"])
    print(f"\n  all_1s vs small_key: t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.01:
        print(f"  *** TIMING DIFFERENCE between high-hw and low-hw keys ***")
    else:
        print(f"  No significant timing difference.")

    # ================================================================
    # TEST 4: Can timing predict MSB?
    # ================================================================
    print(f"\n  TEST 4: Can timing predict MSB of key?")

    msb0_times = []
    msb1_times = []

    for _ in range(N_SAMPLES):
        k = secrets.randbelow(ORDER - 1) + 1
        t = time_multiply(k)
        if (k >> 255) & 1:
            msb1_times.append(t)
        else:
            msb0_times.append(t)

    if msb0_times and msb1_times:
        t_stat, p_val = stats.ttest_ind(msb0_times, msb1_times)
        print(f"  MSB=0: n={len(msb0_times)}, mean={np.mean(msb0_times)/1e6:.3f}ms")
        print(f"  MSB=1: n={len(msb1_times)}, mean={np.mean(msb1_times)/1e6:.3f}ms")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.01:
            print(f"  *** TIMING LEAK: MSB affects timing ***")
        else:
            print(f"  No MSB timing leak.")

    # ================================================================
    # TEST 5: Double-and-add timing pattern
    # ================================================================
    print(f"\n  TEST 5: Does the key's binary representation affect timing?")

    # Keys with long runs of 0s should be faster (fewer adds in double-and-add)
    run0_times = []  # keys with long runs of 0s
    run1_times = []  # keys with long runs of 1s

    for _ in range(100):
        # Long run of 0s: key with few 1-bits
        k_sparse = 0
        for _ in range(10):
            k_sparse |= 1 << secrets.randbelow(256)
        k_sparse = k_sparse % ORDER
        if k_sparse == 0: k_sparse = 1
        t = time_multiply(k_sparse)
        run0_times.append(t)

        # Long run of 1s: key with many 1-bits
        k_dense = (1 << 256) - 1
        for _ in range(10):
            k_dense &= ~(1 << secrets.randbelow(256))
        k_dense = k_dense % ORDER
        if k_dense == 0: k_dense = 1
        t = time_multiply(k_dense)
        run1_times.append(t)

    t_stat, p_val = stats.ttest_ind(run0_times, run1_times)
    print(f"  Sparse keys (low HW): mean={np.mean(run0_times)/1e6:.3f}ms")
    print(f"  Dense keys (high HW):  mean={np.mean(run1_times)/1e6:.3f}ms")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.01:
        print(f"  *** TIMING LEAK: Hamming weight of key affects multiply time ***")
        print(f"  This confirms python-ecdsa uses non-constant-time double-and-add!")
    else:
        print(f"  No significant timing difference.")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  SUMMARY: TIMING SIDE-CHANNEL")
    print(f"{'='*78}")

    leaks_found = []
    if abs(r_hw) > 0.2 and p_hw < 0.05:
        leaks_found.append("Hamming weight")
    if abs(r_bl) > 0.2 and p_bl < 0.05:
        leaks_found.append("Bit length")

    if leaks_found:
        print(f"\n  TIMING LEAKS DETECTED: {', '.join(leaks_found)}")
        print(f"  python-ecdsa MAY use non-constant-time scalar multiplication.")
        print(f"  In a real attack scenario (e.g., TLS server), this could leak")
        print(f"  key bits through network timing measurements.")
        print(f"\n  HOWEVER: this is a known limitation of python-ecdsa.")
        print(f"  Production crypto libraries (OpenSSL, libsecp256k1) use")
        print(f"  constant-time implementations that don't have this leak.")
        print(f"  Bitcoin Core uses libsecp256k1, which is immune.")
    else:
        print(f"\n  No significant timing leaks detected.")
        print(f"  Either python-ecdsa is constant-time, or the variance")
        print(f"  is too high for our measurement resolution.")

    print(f"\n  Key takeaway:")
    print(f"  - Timing side-channels are a REAL threat for bad implementations")
    print(f"  - Bitcoin Core (libsecp256k1) is constant-time: immune")
    print(f"  - This attack requires measuring the SIGNING operation, not public keys")
    print(f"  - Can't attack cold wallets or addresses (need signing access)")
    print("=" * 78)


if __name__ == "__main__":
    main()
