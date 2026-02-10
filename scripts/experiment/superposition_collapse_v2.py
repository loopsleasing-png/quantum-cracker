"""Superposition Collapse Engine v2: Bit-Level Resolution.

Previous: 16 sandboxes x 64 hex positions = 1,024 evaluations
This:     16 sandboxes x 256 bit positions = 4,096 evaluations

Architecture:
  - Precompute 256 bit-basis EC points: P_i = 2^i * G
  - For each bit position (0-255), compute two "remainders":
      R_0 = K                (if bit i is 0, no contribution to subtract)
      R_1 = K - P_i          (if bit i is 1, subtract its contribution)
  - Score both remainders with 5 oracles
  - 16 sandboxes map onto this: each hex value h predicts specific
    bit values at each position via its 4-bit pattern

  Additionally: Successive Interference Cancellation (SIC)
  - Lock the most confident bits, subtract their contributions
  - Re-score remaining bits against the reduced target
  - Iterate until all bits are locked
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import INFINITY, Point

from quantum_cracker.core.key_interface import KeyInput

# EC curve parameters
G = SECP256k1.generator
ORDER = SECP256k1.order
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()

HEX_CHARS = "0123456789abcdef"


def point_negate(pt):
    if pt == INFINITY:
        return INFINITY
    return Point(CURVE, pt.x(), (-pt.y()) % P_FIELD)


def precompute_bit_bases():
    """Precompute P_i = 2^i * G for i = 0..255.

    Bit 0 is the LSB (rightmost), bit 255 is the MSB (leftmost).
    In the hex string, bit 255 is the leftmost bit of the first hex char.
    """
    print("  Precomputing 256 bit-basis EC points...")
    t0 = time.time()

    bases = [None] * 256
    bases[0] = G  # 2^0 * G = G
    for i in range(1, 256):
        bases[i] = bases[i - 1].double()  # 2^i * G = 2 * (2^(i-1) * G)

    print(f"  Done in {time.time()-t0:.1f}s (255 doublings)")
    return bases


def score_remainder(pt):
    """Score an EC point remainder using multiple oracles.

    Returns dict of oracle_name -> score (higher = more likely correct).
    """
    if pt == INFINITY:
        # Point at infinity means all other bits sum to zero.
        # Extremely unlikely but technically the "cleanest" point.
        return {
            "hamming_x": 0,
            "hamming_y": 0,
            "low_bits_x": 10,
            "smoothness": 10,
            "nibble_balance": 0,
        }

    x = pt.x()
    y = pt.y()

    # Oracle 1: x-coordinate Hamming weight distance from 128
    x_hw = bin(x).count("1")
    score_hx = -abs(x_hw - 128)

    # Oracle 2: y-coordinate Hamming weight distance from 128
    y_hw = bin(y).count("1")
    score_hy = -abs(y_hw - 128)

    # Oracle 3: Trailing zero bits in x (smoothness indicator)
    if x == 0:
        trailing = 256
    else:
        trailing = (x & -x).bit_length() - 1
    score_low = trailing

    # Oracle 4: Smoothness -- number of small prime factors of x mod small_prime
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    smooth_score = sum(1 for p in small_primes if x % p == 0)
    score_smooth = smooth_score

    # Oracle 5: Nibble balance -- how uniform is the hex digit distribution
    x_hex = f"{x:064x}"
    nibble_counts = np.array([x_hex.count(c) for c in HEX_CHARS])
    # More uniform = higher score (expected for random EC points)
    score_balance = -np.std(nibble_counts)

    return {
        "hamming_x": score_hx,
        "hamming_y": score_hy,
        "low_bits_x": score_low,
        "smoothness": score_smooth,
        "nibble_balance": score_balance,
    }


def bit_index_to_hex_info(bit_idx):
    """Convert a bit index (0=MSB, 255=LSB) to hex position and sub-bit.

    Returns (hex_pos, sub_bit) where:
      hex_pos: 0-63 (position in hex string)
      sub_bit: 0-3 (bit within the hex digit, 0=MSB of nibble)
    """
    hex_pos = bit_idx // 4
    sub_bit = bit_idx % 4
    return hex_pos, sub_bit


def main():
    print()
    print("=" * 70)
    print("  SUPERPOSITION COLLAPSE v2: BIT-LEVEL")
    print("  16 Sandboxes x 256 Bit Positions = 4,096 evaluations")
    print("=" * 70)

    # Target key
    target_hex = "06d88f2148757a251dd0ea0e6c4584e159a60cfd3f7217c7b0b111adec0efbca"
    target_key = KeyInput(target_hex)
    target_int = target_key.as_int
    actual_bits = target_key.as_bits  # 256 bits, MSB first

    print(f"\n  Target: {target_hex[:16]}...{target_hex[-16:]}")

    # Derive public key
    print("  Deriving public key...")
    t0 = time.time()
    K = G * target_int
    print(f"  Done in {time.time()-t0:.1f}s")

    # Precompute bit bases
    bit_bases = precompute_bit_bases()

    # ================================================================
    # PHASE 1: Independent bit scoring (no communication between bits)
    # ================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: INDEPENDENT BIT SCORING")
    print("  For each bit, score bit=0 vs bit=1 using 5 oracles")
    print("=" * 70)

    oracle_names = ["hamming_x", "hamming_y", "low_bits_x", "smoothness", "nibble_balance"]

    # scores_0[oracle][bit] = score if bit i is 0
    # scores_1[oracle][bit] = score if bit i is 1
    scores_0 = {name: np.zeros(256) for name in oracle_names}
    scores_1 = {name: np.zeros(256) for name in oracle_names}

    print("\n  Scoring 256 bit positions x 2 values x 5 oracles = 2,560 evaluations...")
    t0 = time.time()

    for i in range(256):
        # Bit index i: 0 = MSB (bit 255 in integer), 255 = LSB (bit 0 in integer)
        # In integer representation: bit i corresponds to 2^(255-i)
        power = 255 - i
        P_i = bit_bases[power]

        # R_0 = K (assume bit is 0, no contribution)
        s0 = score_remainder(K)

        # R_1 = K - P_i (assume bit is 1, subtract contribution)
        R_1 = K + point_negate(P_i)
        s1 = score_remainder(R_1)

        for name in oracle_names:
            scores_0[name][i] = s0[name]
            scores_1[name][i] = s1[name]

        if (i + 1) % 64 == 0:
            print(f"    Bit {i+1:3d}/256 complete ({time.time()-t0:.1f}s)")

    total_time = time.time() - t0
    print(f"  All bits scored in {total_time:.1f}s")

    # Per-oracle predictions
    print(f"\n  {'Oracle':18s}  {'Bits Correct':>13s}  {'vs 128':>7s}")
    print(f"  {'-'*18}  {'-'*13}  {'-'*7}")

    oracle_predictions = {}
    for name in oracle_names:
        predicted = np.where(scores_1[name] > scores_0[name], 1, 0)
        correct = sum(p == a for p, a in zip(predicted, actual_bits))
        oracle_predictions[name] = predicted
        print(f"  {name:18s}  {correct:5d}/256      {correct-128:+d}")

    # Ensemble: majority vote across oracles
    all_preds = np.array([oracle_predictions[n] for n in oracle_names])
    ensemble_pred = (np.sum(all_preds, axis=0) > len(oracle_names) / 2).astype(int)
    ensemble_correct = sum(p == a for p, a in zip(ensemble_pred, actual_bits))
    print(f"  {'ENSEMBLE':18s}  {ensemble_correct:5d}/256      {ensemble_correct-128:+d}")

    # ================================================================
    # Map to hex and show 16-sandbox view
    # ================================================================
    print("\n" + "=" * 70)
    print("  16-SANDBOX HEX MAPPING")
    print("  Each sandbox's 4-bit pattern maps to bit predictions")
    print("=" * 70)

    # For each hex position, the ensemble predicted 4 bits -> a hex digit
    predicted_hex_digits = []
    for hp in range(64):
        nibble_bits = ensemble_pred[hp*4 : hp*4 + 4]
        hex_val = int(nibble_bits[0]) * 8 + int(nibble_bits[1]) * 4 + \
                  int(nibble_bits[2]) * 2 + int(nibble_bits[3])
        predicted_hex_digits.append(hex_val)

    actual_hex_digits = [int(c, 16) for c in target_hex]
    hex_correct = sum(p == a for p, a in zip(predicted_hex_digits, actual_hex_digits))

    predicted_hex_str = "".join(HEX_CHARS[h] for h in predicted_hex_digits)
    print(f"\n  Actual:    {target_hex}")
    print(f"  Predicted: {predicted_hex_str}")

    match_str = ""
    for a, p in zip(target_hex, predicted_hex_str):
        match_str += "^" if a == p else " "
    print(f"  Matches:   {match_str}")
    print(f"\n  Hex correct: {hex_correct}/64 ({hex_correct/64*100:.1f}%)")
    print(f"  Bits correct: {ensemble_correct}/256 ({ensemble_correct/256*100:.1f}%)")

    # ================================================================
    # PHASE 2: Successive Interference Cancellation (SIC)
    # ================================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: SUCCESSIVE INTERFERENCE CANCELLATION (SIC)")
    print("  Lock most confident bits, subtract from K, re-score remaining")
    print("=" * 70)

    # Compute per-bit confidence: |score_1 - score_0| averaged across oracles
    confidence = np.zeros(256)
    for name in oracle_names:
        diff = np.abs(scores_1[name] - scores_0[name])
        max_diff = diff.max()
        if max_diff > 0:
            confidence += diff / max_diff

    confidence /= len(oracle_names)

    # Start SIC
    locked = np.full(256, -1, dtype=int)  # -1 = unlocked
    current_K = K  # Remaining target after subtracting locked bits
    sic_rounds = 8
    bits_per_round = 32

    for round_num in range(sic_rounds):
        # Find the most confident unlocked bits
        unlocked_mask = locked == -1
        unlocked_indices = np.where(unlocked_mask)[0]

        if len(unlocked_indices) == 0:
            break

        # Get confidence for unlocked bits
        unlocked_conf = confidence[unlocked_indices]
        n_to_lock = min(bits_per_round, len(unlocked_indices))

        # Pick top-N most confident
        top_idx = np.argsort(unlocked_conf)[-n_to_lock:]
        bits_to_lock = unlocked_indices[top_idx]

        # Score these bits against current_K
        for i in bits_to_lock:
            power = 255 - i
            P_i = bit_bases[power]

            s0 = score_remainder(current_K)
            R_1 = current_K + point_negate(P_i)
            s1 = score_remainder(R_1)

            # Ensemble vote
            vote_1 = sum(1 for name in oracle_names if s1[name] > s0[name])
            predicted_bit = 1 if vote_1 > len(oracle_names) / 2 else 0

            locked[i] = predicted_bit

            # If we predict bit=1, subtract its contribution from current_K
            if predicted_bit == 1:
                current_K = current_K + point_negate(P_i)

        locked_count = np.sum(locked >= 0)
        locked_correct = sum(locked[i] == actual_bits[i]
                             for i in range(256) if locked[i] >= 0)

        print(f"  Round {round_num+1}: locked {locked_count:3d}/256 bits, "
              f"{locked_correct:3d} correct ({locked_correct/locked_count*100:.1f}%)")

    # Final: fill remaining with ensemble predictions
    for i in range(256):
        if locked[i] == -1:
            locked[i] = ensemble_pred[i]

    sic_bits_correct = sum(locked[i] == actual_bits[i] for i in range(256))

    sic_hex_digits = []
    for hp in range(64):
        nibble = locked[hp*4 : hp*4 + 4]
        hex_val = int(nibble[0]) * 8 + int(nibble[1]) * 4 + \
                  int(nibble[2]) * 2 + int(nibble[3])
        sic_hex_digits.append(hex_val)

    sic_hex_correct = sum(p == a for p, a in zip(sic_hex_digits, actual_hex_digits))
    sic_hex_str = "".join(HEX_CHARS[h] for h in sic_hex_digits)

    print(f"\n  SIC Result:")
    print(f"  Actual:    {target_hex}")
    print(f"  Predicted: {sic_hex_str}")

    match_str = ""
    for a, p in zip(target_hex, sic_hex_str):
        match_str += "^" if a == p else " "
    print(f"  Matches:   {match_str}")
    print(f"  Hex correct: {sic_hex_correct}/64 ({sic_hex_correct/64*100:.1f}%)")
    print(f"  Bits correct: {sic_bits_correct}/256 ({sic_bits_correct/256*100:.1f}%)")

    # ================================================================
    # PHASE 3: 16-SANDBOX DIRECT BIT COMPETITION
    # ================================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: 16-SANDBOX COMPETITION PER BIT")
    print("  Each sandbox scores its predicted bit at each position")
    print("  Sandbox with best total score across 4 bits wins the nibble")
    print("=" * 70)

    # For each hex position (64 total), run all 16 hex values
    # Each hex value predicts 4 specific bit values
    # Score = sum of individual bit scores for those 4 predictions

    sandbox_hex_picks = []
    for hp in range(64):
        best_h = 0
        best_score = -np.inf

        for h in range(16):
            h_bits = [(h >> (3 - b)) & 1 for b in range(4)]
            total_score = 0.0

            for sub_b in range(4):
                bit_idx = hp * 4 + sub_b
                predicted_bit = h_bits[sub_b]

                # Sum oracle scores for this prediction
                for name in oracle_names:
                    if predicted_bit == 1:
                        total_score += scores_1[name][bit_idx]
                    else:
                        total_score += scores_0[name][bit_idx]

            if total_score > best_score:
                best_score = total_score
                best_h = h

        sandbox_hex_picks.append(best_h)

    sandbox_hex_str = "".join(HEX_CHARS[h] for h in sandbox_hex_picks)
    sandbox_hex_correct = sum(p == a for p, a in zip(sandbox_hex_picks, actual_hex_digits))
    sandbox_bits_correct = sum(
        int(a) == int(b)
        for a, b in zip(f"{int(sandbox_hex_str, 16):0256b}",
                        f"{int(target_hex, 16):0256b}")
    )

    print(f"\n  Actual:    {target_hex}")
    print(f"  Predicted: {sandbox_hex_str}")

    match_str = ""
    for a, p in zip(target_hex, sandbox_hex_str):
        match_str += "^" if a == p else " "
    print(f"  Matches:   {match_str}")
    print(f"  Hex correct: {sandbox_hex_correct}/64 ({sandbox_hex_correct/64*100:.1f}%)")
    print(f"  Bits correct: {sandbox_bits_correct}/256 ({sandbox_bits_correct/256*100:.1f}%)")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY: ALL METHODS")
    print("=" * 70)

    print(f"\n  {'Method':35s}  {'Hex':>7s}  {'Bits':>9s}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*9}")
    print(f"  {'Expected (random)':35s}  {'4/64':>7s}  {'128/256':>9s}")
    for name in oracle_names:
        pred = oracle_predictions[name]
        bc = sum(p == a for p, a in zip(pred, actual_bits))
        hp = []
        for i in range(64):
            nb = pred[i*4:i*4+4]
            hv = int(nb[0])*8 + int(nb[1])*4 + int(nb[2])*2 + int(nb[3])
            hp.append(hv)
        hc = sum(p == a for p, a in zip(hp, actual_hex_digits))
        print(f"  {name:35s}  {hc:3d}/64  {bc:5d}/256")
    print(f"  {'Ensemble (majority vote)':35s}  {hex_correct:3d}/64  {ensemble_correct:5d}/256")
    print(f"  {'SIC (iterative cancellation)':35s}  {sic_hex_correct:3d}/64  {sic_bits_correct:5d}/256")
    print(f"  {'16-Sandbox competition':35s}  {sandbox_hex_correct:3d}/64  {sandbox_bits_correct:5d}/256")

    # ================================================================
    # THE PHYSICS
    # ================================================================
    print("\n" + "=" * 70)
    print("  THE PHYSICS: WHY PER-BIT FAILS TOO")
    print("=" * 70)
    print("""
  Scaling from 64 to 256 positions doesn't help because the
  fundamental problem is the SCORING, not the resolution.

  At hex level:  K - h * 16^p * G = valid point (no per-hex feedback)
  At bit level:  K - b * 2^i * G  = valid point (no per-bit feedback)

  The EC group has no "texture" -- every point looks the same.
  Hamming weight, entropy, smoothness, trailing zeros of x-coordinates
  are all uniformly distributed for random EC points.

  To get per-position feedback, you need one of:
    1. A quantum computer (Grover's oracle marks correct state)
    2. A mathematical break of ECDLP (none known)
    3. A side channel (timing, power, EM radiation from real hardware)

  The sandboxes are isolated and score independently.
  But without a scoring function that can distinguish correct from wrong,
  isolation doesn't help -- 16 blind sandboxes are 16x as blind.
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
