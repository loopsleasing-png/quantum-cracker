"""Superposition Collapse v3: 256 Oracles x 256 Bit Positions.

Previous: 5 oracles -- all returned ~50%
This:     256 oracles -- every mathematical property of an EC point
          we can think of, searching for ANY signal at all.

Oracle categories (256 total):
  [  0- 31] Bit statistics of x-coordinate (32)
  [ 32- 63] Bit statistics of y-coordinate (32)
  [ 64- 95] Nibble/hex statistics of x (32)
  [ 96-127] Nibble/hex statistics of y (32)
  [128-159] Modular arithmetic: x mod primes, y mod primes (32)
  [160-191] Cross x-y relationships (32)
  [192-223] Positional: specific bit values of x and y (32)
  [224-255] Spectral: DFT of bit sequences of x and y (32)

If ANY of these 256 oracles consistently scores above random,
there's a signal in EC point structure we can exploit.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import INFINITY, Point

from quantum_cracker.core.key_interface import KeyInput

G = SECP256k1.generator
ORDER = SECP256k1.order
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()

FIRST_32_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
]

ORACLE_NAMES = []


def _name(cat, idx, desc):
    name = f"{cat:02d}_{idx:02d}_{desc}"
    ORACLE_NAMES.append(name)
    return len(ORACLE_NAMES) - 1


def build_oracle_index():
    """Register all 256 oracles and return name list."""
    ORACLE_NAMES.clear()

    # Category 0: Bit stats of x (indices 0-31)
    _name(0, 0, "x_hamming_weight")
    for q in range(4):
        _name(0, 1 + q, f"x_hw_quarter_{q}")
    for c in range(16):
        _name(0, 5 + c, f"x_hw_chunk16_{c}")
    _name(0, 21, "x_trailing_zeros")
    _name(0, 22, "x_leading_zeros")
    _name(0, 23, "x_longest_run_1")
    _name(0, 24, "x_longest_run_0")
    _name(0, 25, "x_parity")
    _name(0, 26, "x_transitions")
    _name(0, 27, "x_autocorr_lag1")
    _name(0, 28, "x_autocorr_lag2")
    _name(0, 29, "x_autocorr_lag4")
    _name(0, 30, "x_autocorr_lag8")
    _name(0, 31, "x_block_xor_16")

    # Category 1: Bit stats of y (indices 32-63)
    _name(1, 0, "y_hamming_weight")
    for q in range(4):
        _name(1, 1 + q, f"y_hw_quarter_{q}")
    for c in range(16):
        _name(1, 5 + c, f"y_hw_chunk16_{c}")
    _name(1, 21, "y_trailing_zeros")
    _name(1, 22, "y_leading_zeros")
    _name(1, 23, "y_longest_run_1")
    _name(1, 24, "y_longest_run_0")
    _name(1, 25, "y_parity")
    _name(1, 26, "y_transitions")
    _name(1, 27, "y_autocorr_lag1")
    _name(1, 28, "y_autocorr_lag2")
    _name(1, 29, "y_autocorr_lag4")
    _name(1, 30, "y_autocorr_lag8")
    _name(1, 31, "y_block_xor_16")

    # Category 2: Nibble stats of x (indices 64-95)
    for n in range(16):
        _name(2, n, f"x_nibble_count_{n:x}")
    _name(2, 16, "x_entropy")
    _name(2, 17, "x_max_nibble_freq")
    _name(2, 18, "x_min_nibble_freq")
    _name(2, 19, "x_unique_nibbles")
    _name(2, 20, "x_nibble_sum")
    _name(2, 21, "x_nibble_std")
    _name(2, 22, "x_longest_nibble_run")
    _name(2, 23, "x_nibble_transitions")
    _name(2, 24, "x_even_nibble_count")
    _name(2, 25, "x_high_nibble_count")
    _name(2, 26, "x_nibble_range")
    _name(2, 27, "x_nibble_median")
    _name(2, 28, "x_ascending_pairs")
    _name(2, 29, "x_palindrome_score")
    _name(2, 30, "x_nibble_mode")
    _name(2, 31, "x_nibble_autocorr1")

    # Category 3: Nibble stats of y (indices 96-127)
    for n in range(16):
        _name(3, n, f"y_nibble_count_{n:x}")
    _name(3, 16, "y_entropy")
    _name(3, 17, "y_max_nibble_freq")
    _name(3, 18, "y_min_nibble_freq")
    _name(3, 19, "y_unique_nibbles")
    _name(3, 20, "y_nibble_sum")
    _name(3, 21, "y_nibble_std")
    _name(3, 22, "y_longest_nibble_run")
    _name(3, 23, "y_nibble_transitions")
    _name(3, 24, "y_even_nibble_count")
    _name(3, 25, "y_high_nibble_count")
    _name(3, 26, "y_nibble_range")
    _name(3, 27, "y_nibble_median")
    _name(3, 28, "y_ascending_pairs")
    _name(3, 29, "y_palindrome_score")
    _name(3, 30, "y_nibble_mode")
    _name(3, 31, "y_nibble_autocorr1")

    # Category 4: Modular arithmetic (indices 128-159)
    for i in range(16):
        _name(4, i, f"x_mod_{FIRST_32_PRIMES[i]}")
    for i in range(16):
        _name(4, 16 + i, f"y_mod_{FIRST_32_PRIMES[i]}")

    # Category 5: Cross x-y (indices 160-191)
    _name(5, 0, "xy_hamming_dist")
    _name(5, 1, "xor_xy_hw")
    _name(5, 2, "and_xy_hw")
    _name(5, 3, "or_xy_hw")
    _name(5, 4, "x_plus_y_hw")
    _name(5, 5, "x_minus_y_hw")
    _name(5, 6, "x_times_y_hw")
    _name(5, 7, "common_trailing_zeros")
    _name(5, 8, "common_leading_zeros")
    _name(5, 9, "x_gt_y")
    _name(5, 10, "abs_diff_hw")
    _name(5, 11, "xor_nibble_entropy")
    _name(5, 12, "sum_mod_order_hw")
    _name(5, 13, "diff_hamming_weight")
    _name(5, 14, "xy_nibble_correlation")
    _name(5, 15, "xor_trailing_zeros")
    for c in range(16):
        _name(5, 16 + c, f"xor_xy_chunk16_{c}_hw")

    # Category 6: Positional bits (indices 192-223)
    positions = list(range(0, 256, 8))  # every 8th bit = 32 positions
    for i, pos in enumerate(positions[:16]):
        _name(6, i, f"x_bit_{pos}")
    for i, pos in enumerate(positions[:16]):
        _name(6, 16 + i, f"y_bit_{pos}")

    # Category 7: Spectral (indices 224-255)
    for f in range(16):
        _name(7, f, f"x_dft_mag_{f}")
    for f in range(16):
        _name(7, 16 + f, f"y_dft_mag_{f}")

    assert len(ORACLE_NAMES) == 256, f"Expected 256 oracles, got {len(ORACLE_NAMES)}"
    return ORACLE_NAMES


def int_to_bits_array(val, nbits=256):
    """Convert integer to numpy bit array (MSB first)."""
    return np.array([(val >> (nbits - 1 - i)) & 1 for i in range(nbits)], dtype=np.int8)


def longest_run(bits, target):
    """Longest consecutive run of target value in bit array."""
    max_run = 0
    current = 0
    for b in bits:
        if b == target:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def bit_transitions(bits):
    """Count number of 0->1 or 1->0 transitions."""
    return sum(bits[i] != bits[i + 1] for i in range(len(bits) - 1))


def autocorrelation(bits, lag):
    """Autocorrelation of bit sequence at given lag."""
    n = len(bits)
    if lag >= n:
        return 0
    mean = np.mean(bits)
    shifted = bits[lag:]
    original = bits[:n - lag]
    cov = np.mean((original - mean) * (shifted - mean))
    var = np.var(bits)
    return cov / var if var > 0 else 0


def nibble_array(hex_str):
    """Convert hex string to array of nibble values."""
    return np.array([int(c, 16) for c in hex_str], dtype=np.int8)


def nibble_longest_run(nibbles):
    """Longest consecutive run of same nibble."""
    max_run = 1
    current = 1
    for i in range(1, len(nibbles)):
        if nibbles[i] == nibbles[i - 1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run


def evaluate_all_256_oracles(x, y):
    """Evaluate all 256 oracles on an EC point (x, y).

    Returns: (256,) float array of oracle scores.
    """
    scores = np.zeros(256, dtype=np.float64)

    x_bits = int_to_bits_array(x)
    y_bits = int_to_bits_array(y)
    x_hex = f"{x:064x}"
    y_hex = f"{y:064x}"
    x_nibs = nibble_array(x_hex)
    y_nibs = nibble_array(y_hex)

    # ================================================================
    # Category 0: Bit stats of x (0-31)
    # ================================================================
    idx = 0
    x_hw = int(x_bits.sum())
    scores[idx] = -abs(x_hw - 128); idx += 1

    for q in range(4):
        chunk = x_bits[q * 64:(q + 1) * 64]
        scores[idx] = -abs(int(chunk.sum()) - 32); idx += 1

    for c in range(16):
        chunk = x_bits[c * 16:(c + 1) * 16]
        scores[idx] = -abs(int(chunk.sum()) - 8); idx += 1

    # Trailing zeros
    if x == 0:
        scores[idx] = 256
    else:
        scores[idx] = (x & -x).bit_length() - 1
    idx += 1

    # Leading zeros
    scores[idx] = 256 - x.bit_length(); idx += 1

    scores[idx] = longest_run(x_bits, 1); idx += 1
    scores[idx] = longest_run(x_bits, 0); idx += 1
    scores[idx] = x_hw % 2; idx += 1
    scores[idx] = bit_transitions(x_bits); idx += 1
    scores[idx] = autocorrelation(x_bits.astype(float), 1); idx += 1
    scores[idx] = autocorrelation(x_bits.astype(float), 2); idx += 1
    scores[idx] = autocorrelation(x_bits.astype(float), 4); idx += 1
    scores[idx] = autocorrelation(x_bits.astype(float), 8); idx += 1

    # Block XOR: XOR all 16-bit blocks together
    block_xor = 0
    for c in range(16):
        chunk_val = int.from_bytes(x.to_bytes(32, "big")[c * 2:(c + 1) * 2], "big")
        block_xor ^= chunk_val
    scores[idx] = bin(block_xor).count("1"); idx += 1

    # ================================================================
    # Category 1: Bit stats of y (32-63)
    # ================================================================
    y_hw = int(y_bits.sum())
    scores[idx] = -abs(y_hw - 128); idx += 1

    for q in range(4):
        chunk = y_bits[q * 64:(q + 1) * 64]
        scores[idx] = -abs(int(chunk.sum()) - 32); idx += 1

    for c in range(16):
        chunk = y_bits[c * 16:(c + 1) * 16]
        scores[idx] = -abs(int(chunk.sum()) - 8); idx += 1

    if y == 0:
        scores[idx] = 256
    else:
        scores[idx] = (y & -y).bit_length() - 1
    idx += 1

    scores[idx] = 256 - y.bit_length(); idx += 1
    scores[idx] = longest_run(y_bits, 1); idx += 1
    scores[idx] = longest_run(y_bits, 0); idx += 1
    scores[idx] = y_hw % 2; idx += 1
    scores[idx] = bit_transitions(y_bits); idx += 1
    scores[idx] = autocorrelation(y_bits.astype(float), 1); idx += 1
    scores[idx] = autocorrelation(y_bits.astype(float), 2); idx += 1
    scores[idx] = autocorrelation(y_bits.astype(float), 4); idx += 1
    scores[idx] = autocorrelation(y_bits.astype(float), 8); idx += 1

    block_xor = 0
    for c in range(16):
        chunk_val = int.from_bytes(y.to_bytes(32, "big")[c * 2:(c + 1) * 2], "big")
        block_xor ^= chunk_val
    scores[idx] = bin(block_xor).count("1"); idx += 1

    # ================================================================
    # Category 2: Nibble stats of x (64-95)
    # ================================================================
    for n in range(16):
        scores[idx] = np.sum(x_nibs == n); idx += 1

    freq = np.bincount(x_nibs, minlength=16).astype(float) / 64
    entropy = -np.sum(freq[freq > 0] * np.log2(freq[freq > 0]))
    scores[idx] = entropy; idx += 1
    scores[idx] = freq.max(); idx += 1
    scores[idx] = -freq[freq > 0].min(); idx += 1
    scores[idx] = np.sum(freq > 0); idx += 1
    scores[idx] = float(x_nibs.sum()); idx += 1
    scores[idx] = -float(np.std(x_nibs)); idx += 1
    scores[idx] = nibble_longest_run(x_nibs); idx += 1
    scores[idx] = sum(x_nibs[i] != x_nibs[i + 1] for i in range(63)); idx += 1
    scores[idx] = np.sum(x_nibs % 2 == 0); idx += 1
    scores[idx] = np.sum(x_nibs >= 8); idx += 1
    scores[idx] = int(x_nibs.max()) - int(x_nibs.min()); idx += 1
    scores[idx] = float(np.median(x_nibs)); idx += 1
    scores[idx] = sum(x_nibs[i] < x_nibs[i + 1] for i in range(63)); idx += 1
    # Palindrome: compare first 32 nibbles to reversed last 32
    scores[idx] = sum(x_nibs[i] == x_nibs[63 - i] for i in range(32)); idx += 1
    # Mode
    counts = np.bincount(x_nibs, minlength=16)
    scores[idx] = float(np.argmax(counts)); idx += 1
    # Nibble autocorrelation lag 1
    x_nibs_f = x_nibs.astype(float)
    m = x_nibs_f.mean()
    v = x_nibs_f.var()
    if v > 0:
        scores[idx] = float(np.mean((x_nibs_f[:-1] - m) * (x_nibs_f[1:] - m)) / v)
    idx += 1

    # ================================================================
    # Category 3: Nibble stats of y (96-127)
    # ================================================================
    for n in range(16):
        scores[idx] = np.sum(y_nibs == n); idx += 1

    freq = np.bincount(y_nibs, minlength=16).astype(float) / 64
    entropy = -np.sum(freq[freq > 0] * np.log2(freq[freq > 0]))
    scores[idx] = entropy; idx += 1
    scores[idx] = freq.max(); idx += 1
    scores[idx] = -freq[freq > 0].min(); idx += 1
    scores[idx] = np.sum(freq > 0); idx += 1
    scores[idx] = float(y_nibs.sum()); idx += 1
    scores[idx] = -float(np.std(y_nibs)); idx += 1
    scores[idx] = nibble_longest_run(y_nibs); idx += 1
    scores[idx] = sum(y_nibs[i] != y_nibs[i + 1] for i in range(63)); idx += 1
    scores[idx] = np.sum(y_nibs % 2 == 0); idx += 1
    scores[idx] = np.sum(y_nibs >= 8); idx += 1
    scores[idx] = int(y_nibs.max()) - int(y_nibs.min()); idx += 1
    scores[idx] = float(np.median(y_nibs)); idx += 1
    scores[idx] = sum(y_nibs[i] < y_nibs[i + 1] for i in range(63)); idx += 1
    scores[idx] = sum(y_nibs[i] == y_nibs[63 - i] for i in range(32)); idx += 1
    counts = np.bincount(y_nibs, minlength=16)
    scores[idx] = float(np.argmax(counts)); idx += 1
    y_nibs_f = y_nibs.astype(float)
    m = y_nibs_f.mean()
    v = y_nibs_f.var()
    if v > 0:
        scores[idx] = float(np.mean((y_nibs_f[:-1] - m) * (y_nibs_f[1:] - m)) / v)
    idx += 1

    # ================================================================
    # Category 4: Modular arithmetic (128-159)
    # ================================================================
    for i in range(16):
        scores[idx] = -(x % FIRST_32_PRIMES[i]); idx += 1
    for i in range(16):
        scores[idx] = -(y % FIRST_32_PRIMES[i]); idx += 1

    # ================================================================
    # Category 5: Cross x-y (160-191)
    # ================================================================
    xor_xy = x ^ y
    and_xy = x & y
    or_xy = x | y

    scores[idx] = -abs(bin(xor_xy).count("1") - 128); idx += 1
    scores[idx] = bin(xor_xy).count("1"); idx += 1
    scores[idx] = bin(and_xy).count("1"); idx += 1
    scores[idx] = bin(or_xy).count("1"); idx += 1
    scores[idx] = -abs(bin((x + y) % (2**256)).count("1") - 128); idx += 1
    scores[idx] = -abs(bin((x - y) % (2**256)).count("1") - 128); idx += 1
    scores[idx] = -abs(bin((x * y) % P_FIELD).count("1") - 128); idx += 1

    # Common trailing zeros
    if (x | y) == 0:
        scores[idx] = 256
    elif (x & y) == 0:
        scores[idx] = 0
    else:
        scores[idx] = ((x & y) & -(x & y)).bit_length() - 1
    idx += 1

    # Common leading zeros
    scores[idx] = 256 - max(x.bit_length(), y.bit_length()); idx += 1

    scores[idx] = 1.0 if x > y else -1.0; idx += 1
    scores[idx] = -abs(x_hw - y_hw); idx += 1

    # XOR nibble entropy
    xor_hex = f"{xor_xy:064x}"
    xor_nibs = nibble_array(xor_hex)
    xor_freq = np.bincount(xor_nibs, minlength=16).astype(float) / 64
    xor_ent = -np.sum(xor_freq[xor_freq > 0] * np.log2(xor_freq[xor_freq > 0]))
    scores[idx] = xor_ent; idx += 1

    scores[idx] = -abs(bin((x + y) % ORDER).count("1") - 128); idx += 1
    scores[idx] = abs(x_hw - y_hw); idx += 1

    # Nibble correlation
    corr = np.corrcoef(x_nibs.astype(float), y_nibs.astype(float))[0, 1]
    scores[idx] = corr if not np.isnan(corr) else 0; idx += 1

    if xor_xy == 0:
        scores[idx] = 256
    else:
        scores[idx] = (xor_xy & -xor_xy).bit_length() - 1
    idx += 1

    # XOR chunk hamming weights
    xor_bits = int_to_bits_array(xor_xy)
    for c in range(16):
        chunk = xor_bits[c * 16:(c + 1) * 16]
        scores[idx] = -abs(int(chunk.sum()) - 8); idx += 1

    # ================================================================
    # Category 6: Positional bits (192-223)
    # ================================================================
    positions = list(range(0, 256, 8))  # 32 positions
    for pos in positions[:16]:
        scores[idx] = float((x >> (255 - pos)) & 1); idx += 1
    for pos in positions[:16]:
        scores[idx] = float((y >> (255 - pos)) & 1); idx += 1

    # ================================================================
    # Category 7: Spectral (224-255)
    # ================================================================
    # DFT of x bit sequence, first 16 frequency magnitudes
    x_signal = x_bits.astype(float) * 2 - 1  # map to +/-1
    x_fft = np.abs(np.fft.rfft(x_signal))
    for f in range(16):
        scores[idx] = x_fft[f + 1] if f + 1 < len(x_fft) else 0; idx += 1

    y_signal = y_bits.astype(float) * 2 - 1
    y_fft = np.abs(np.fft.rfft(y_signal))
    for f in range(16):
        scores[idx] = y_fft[f + 1] if f + 1 < len(y_fft) else 0; idx += 1

    assert idx == 256, f"Oracle count mismatch: {idx}"
    return scores


def main():
    print()
    print("=" * 70)
    print("  SUPERPOSITION COLLAPSE v3: 256 ORACLES")
    print("  256 oracles x 256 bit positions = 65,536 scores per candidate")
    print("=" * 70)

    target_hex = "06d88f2148757a251dd0ea0e6c4584e159a60cfd3f7217c7b0b111adec0efbca"
    target_key = KeyInput(target_hex)
    target_int = target_key.as_int
    actual_bits = np.array(target_key.as_bits, dtype=np.int8)

    print(f"\n  Target: {target_hex[:16]}...{target_hex[-16:]}")

    # Derive public key
    print("  Deriving public key...")
    K = G * target_int
    Kx, Ky = K.x(), K.y()

    # Precompute bit bases
    print("  Precomputing 256 bit-basis EC points...")
    bit_bases = [None] * 256
    bit_bases[0] = G
    for i in range(1, 256):
        bit_bases[i] = bit_bases[i - 1].double()

    # Register oracle names
    names = build_oracle_index()
    print(f"  Registered {len(names)} oracles across 8 categories")

    # ================================================================
    # SCORE ALL 256 BITS x 2 VALUES x 256 ORACLES
    # ================================================================
    print("\n" + "=" * 70)
    print("  SCORING: 256 bits x 2 candidates x 256 oracles")
    print(f"  Total evaluations: {256 * 2 * 256:,} = 131,072")
    print("=" * 70)

    # scores[oracle_idx, bit_idx, 0_or_1]
    all_scores = np.zeros((256, 256, 2), dtype=np.float64)

    # Score for K itself (used for bit=0 at every position)
    print("\n  Scoring base point K (used for all bit=0 cases)...")
    t0 = time.time()
    base_scores = evaluate_all_256_oracles(Kx, Ky)
    print(f"  Base scored in {time.time()-t0:.2f}s")

    # For bit=0 at any position, the remainder is K itself
    for bit_idx in range(256):
        all_scores[:, bit_idx, 0] = base_scores

    # For bit=1 at each position, compute K - P_i
    print("  Scoring 256 remainders (K - 2^i * G)...")
    t0 = time.time()
    for bit_idx in range(256):
        power = 255 - bit_idx  # MSB first
        P_i = bit_bases[power]
        neg_P_i = Point(CURVE, P_i.x(), (-P_i.y()) % P_FIELD)
        R = K + neg_P_i

        if R == INFINITY:
            all_scores[:, bit_idx, 1] = 0
        else:
            all_scores[:, bit_idx, 1] = evaluate_all_256_oracles(R.x(), R.y())

        if (bit_idx + 1) % 64 == 0:
            elapsed = time.time() - t0
            rate = (bit_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (256 - bit_idx - 1) / rate if rate > 0 else 0
            print(f"    Bit {bit_idx+1:3d}/256 ({elapsed:.1f}s, ETA {eta:.0f}s)")

    total_time = time.time() - t0
    print(f"  All 256 remainders scored in {total_time:.1f}s")

    # ================================================================
    # PER-ORACLE RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("  PER-ORACLE BIT PREDICTION ACCURACY")
    print("=" * 70)

    oracle_accuracies = np.zeros(256)
    oracle_predictions = np.zeros((256, 256), dtype=np.int8)

    for oi in range(256):
        # For each bit, pick the candidate (0 or 1) with higher oracle score
        pred = np.where(all_scores[oi, :, 1] > all_scores[oi, :, 0], 1, 0).astype(np.int8)
        correct = np.sum(pred == actual_bits)
        oracle_accuracies[oi] = correct
        oracle_predictions[oi] = pred

    # Sort by accuracy
    sorted_idx = np.argsort(oracle_accuracies)[::-1]

    print(f"\n  Top 20 oracles:")
    print(f"  {'Rank':>4s}  {'Oracle':40s}  {'Bits':>9s}  {'vs 128':>7s}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*9}  {'-'*7}")
    for rank, oi in enumerate(sorted_idx[:20]):
        acc = int(oracle_accuracies[oi])
        print(f"  {rank+1:4d}  {names[oi]:40s}  {acc:5d}/256  {acc-128:+d}")

    print(f"\n  Bottom 5 oracles:")
    for rank, oi in enumerate(sorted_idx[-5:]):
        acc = int(oracle_accuracies[oi])
        print(f"  {256-4+rank:4d}  {names[oi]:40s}  {acc:5d}/256  {acc-128:+d}")

    # Distribution
    print(f"\n  Accuracy distribution:")
    bins = [(0, 110), (110, 120), (120, 125), (125, 131),
            (131, 137), (137, 145), (145, 160), (160, 257)]
    for lo, hi in bins:
        count = np.sum((oracle_accuracies >= lo) & (oracle_accuracies < hi))
        bar = "#" * count
        print(f"  {lo:3d}-{hi-1:3d}: {count:3d} {bar}")

    # ================================================================
    # ENSEMBLE METHODS
    # ================================================================
    print("\n" + "=" * 70)
    print("  ENSEMBLE METHODS")
    print("=" * 70)

    # Method 1: Majority vote across all 256 oracles
    vote_sum = oracle_predictions.sum(axis=0)  # (256,) -- how many oracles predict 1
    majority_pred = np.where(vote_sum > 128, 1, 0).astype(np.int8)
    majority_correct = int(np.sum(majority_pred == actual_bits))
    print(f"\n  All-256 majority vote:     {majority_correct}/256 bits ({majority_correct-128:+d})")

    # Method 2: Top-10 oracles majority vote
    top10_preds = oracle_predictions[sorted_idx[:10]]
    top10_vote = top10_preds.sum(axis=0)
    top10_pred = np.where(top10_vote > 5, 1, 0).astype(np.int8)
    top10_correct = int(np.sum(top10_pred == actual_bits))
    print(f"  Top-10 majority vote:      {top10_correct}/256 bits ({top10_correct-128:+d})")

    # Method 3: Top-50 oracles majority vote
    top50_preds = oracle_predictions[sorted_idx[:50]]
    top50_vote = top50_preds.sum(axis=0)
    top50_pred = np.where(top50_vote > 25, 1, 0).astype(np.int8)
    top50_correct = int(np.sum(top50_pred == actual_bits))
    print(f"  Top-50 majority vote:      {top50_correct}/256 bits ({top50_correct-128:+d})")

    # Method 4: Weighted vote (weight by accuracy)
    weights = oracle_accuracies - 128  # positive = above random
    weighted_sum = np.zeros(256)
    for oi in range(256):
        if weights[oi] > 0:
            weighted_sum += weights[oi] * (oracle_predictions[oi] * 2 - 1)
    weighted_pred = np.where(weighted_sum > 0, 1, 0).astype(np.int8)
    weighted_correct = int(np.sum(weighted_pred == actual_bits))
    print(f"  Accuracy-weighted vote:    {weighted_correct}/256 bits ({weighted_correct-128:+d})")

    # Method 5: Per-category best
    print(f"\n  Per-category best oracle:")
    categories = [
        ("Bit stats x", 0, 32),
        ("Bit stats y", 32, 64),
        ("Nibble stats x", 64, 96),
        ("Nibble stats y", 96, 128),
        ("Modular arith", 128, 160),
        ("Cross x-y", 160, 192),
        ("Positional", 192, 224),
        ("Spectral", 224, 256),
    ]
    for cat_name, lo, hi in categories:
        cat_acc = oracle_accuracies[lo:hi]
        best_in_cat = lo + np.argmax(cat_acc)
        acc = int(oracle_accuracies[best_in_cat])
        print(f"    {cat_name:18s}: {names[best_in_cat]:35s} {acc}/256 ({acc-128:+d})")

    # ================================================================
    # HEX ASSEMBLY
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINAL KEY ASSEMBLY")
    print("=" * 70)

    best_pred = weighted_pred  # use the best ensemble
    best_label = "accuracy-weighted"
    best_bits = weighted_correct

    pred_hex_digits = []
    actual_hex_digits = [int(c, 16) for c in target_hex]
    for hp in range(64):
        nibble = best_pred[hp * 4: hp * 4 + 4]
        hv = int(nibble[0]) * 8 + int(nibble[1]) * 4 + int(nibble[2]) * 2 + int(nibble[3])
        pred_hex_digits.append(hv)

    hex_correct = sum(p == a for p, a in zip(pred_hex_digits, actual_hex_digits))
    pred_hex_str = "".join("0123456789abcdef"[h] for h in pred_hex_digits)

    print(f"\n  Method: {best_label}")
    print(f"  Actual:    {target_hex}")
    print(f"  Predicted: {pred_hex_str}")
    match_str = "".join("^" if a == p else " " for a, p in zip(target_hex, pred_hex_str))
    print(f"  Matches:   {match_str}")
    print(f"\n  Hex correct: {hex_correct}/64 ({hex_correct/64*100:.1f}%)")
    print(f"  Bits correct: {best_bits}/256 ({best_bits/256*100:.1f}%)")
    print(f"  Expected:     4/64 hex, 128/256 bits")

    # ================================================================
    # STATISTICAL SIGNIFICANCE
    # ================================================================
    print("\n" + "=" * 70)
    print("  STATISTICAL SIGNIFICANCE TEST")
    print("=" * 70)

    # Under null hypothesis (random), each oracle gets 128 +/- 8 bits correct
    # (binomial with n=256, p=0.5 -> mean=128, std=8)
    # 95% CI: [112, 144], 99% CI: [107, 149]
    mean_acc = np.mean(oracle_accuracies)
    std_acc = np.std(oracle_accuracies)
    max_acc = np.max(oracle_accuracies)
    min_acc = np.min(oracle_accuracies)

    above_144 = np.sum(oracle_accuracies > 144)
    above_149 = np.sum(oracle_accuracies > 149)
    below_112 = np.sum(oracle_accuracies < 112)

    print(f"\n  256 oracles, each predicting 256 bits:")
    print(f"  Mean accuracy: {mean_acc:.1f}/256")
    print(f"  Std:           {std_acc:.1f}")
    print(f"  Range:         [{int(min_acc)}, {int(max_acc)}]")
    print(f"\n  Under random (binomial n=256, p=0.5):")
    print(f"  Expected mean: 128.0, Expected std: 8.0")
    print(f"  95% CI: [112, 144], 99% CI: [107, 149]")
    print(f"\n  Oracles above 95% CI (>144): {above_144}/256")
    print(f"  Oracles above 99% CI (>149): {above_149}/256")
    print(f"  Oracles below 95% CI (<112): {below_112}/256")

    # With 256 independent tests at 5% significance, expected false positives: 12.8
    print(f"\n  Expected false positives at 5%: 12.8 (Bonferroni: need >{144 + 2.5:.0f})")
    if above_149 > 0:
        print(f"\n  *** {above_149} oracle(s) above 99% CI -- investigating ***")
        for oi in range(256):
            if oracle_accuracies[oi] > 149:
                print(f"      {names[oi]}: {int(oracle_accuracies[oi])}/256")
    else:
        print(f"\n  No oracle exceeds 99% CI. No signal detected.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
