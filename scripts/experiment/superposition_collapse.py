"""Superposition Collapse Engine.

Mathematical framework:
  - Constraint Satisfaction Problem (CSP): 64 variables, 16 values each
  - Grover's Algorithm analog: superposition -> oracle -> collapse
  - Parallel Independent Scoring with Late Fusion (the "C compiler" model)

Architecture:
  16 Sandboxes (one per hex value 0-f), fully isolated
  Each sandbox "blares its sound" across all 64 positions
  Each returns a 64-element resonance score vector
  A Linker assembles 16 vectors into a 64x16 matrix and collapses

The "resonance" question: what does each sandbox score against?
  We test 5 independent oracles to find which (if any) provides signal.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import INFINITY, Point

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.utils.constants import GRID_SIZE, NUM_THREADS
from quantum_cracker.utils.math_helpers import build_qr_sh_basis

# EC curve parameters
G = SECP256k1.generator
ORDER = SECP256k1.order
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()

HEX_CHARS = "0123456789abcdef"


def point_negate(pt):
    """Negate an EC point: (x, y) -> (x, -y mod p)."""
    if pt == INFINITY:
        return INFINITY
    return Point(CURVE, pt.x(), (-pt.y()) % P_FIELD)


def point_subtract(a, b):
    """Subtract EC points: a - b."""
    return a + point_negate(b)


def precompute_position_bases():
    """Precompute P_p = 16^(63-p) * G for each hex position p = 0..63.

    Position 0 is the leftmost (most significant) hex digit.
    So position 0 contributes h * 16^63 * G, position 63 contributes h * G.
    """
    print("  Precomputing EC position bases (64 points)...")
    t0 = time.time()

    # Start from the least significant position
    bases = [None] * 64
    bases[63] = G  # position 63 (rightmost) = 16^0 * G = G

    # P_{p-1} = 16 * P_p (moving left = more significant = higher power)
    for p in range(62, -1, -1):
        # 16 * P = 4 doublings
        pt = bases[p + 1]
        for _ in range(4):
            pt = pt.double()
        bases[p] = pt

    print(f"  Done in {time.time()-t0:.1f}s")
    return bases


def precompute_hex_multiples(position_bases):
    """For each position p and hex value h, compute h * P_p.

    Returns: dict[(p, h)] -> EC Point
    """
    print("  Precomputing hex multiples (1024 points)...")
    t0 = time.time()

    multiples = {}
    for p in range(64):
        base = position_bases[p]
        multiples[(p, 0)] = INFINITY
        multiples[(p, 1)] = base
        for h in range(2, 16):
            multiples[(p, h)] = multiples[(p, h - 1)] + base

    print(f"  Done in {time.time()-t0:.1f}s")
    return multiples


def build_harmonic_signatures():
    """Build the 16 harmonic signatures (one per hex digit).

    Each hex digit maps to 4 bits, which map to 4 SH coefficient signs.
    """
    sigs = {}
    for h in range(16):
        bits = [(h >> (3 - b)) & 1 for b in range(4)]
        coeffs = [2.0 * b - 1.0 for b in bits]
        sigs[h] = np.array(coeffs, dtype=np.float64)
    return sigs


class Sandbox:
    """One of 16 isolated sandboxes. Each owns a single hex value.

    The sandbox knows:
      - Its hex value (0-15)
      - The target public key point
      - The precomputed EC points
      - The target public key's harmonic field (derived from pub_x, pub_y)

    It does NOT know:
      - The private key
      - What the other 15 sandboxes are doing
      - Which positions are "taken" by other hex values
    """

    def __init__(self, hex_value, target_pub, position_bases, hex_multiples,
                 pub_x_field, pub_y_field, harmonic_sigs):
        self.h = hex_value
        self.label = HEX_CHARS[hex_value]
        self.target_pub = target_pub
        self.position_bases = position_bases
        self.hex_multiples = hex_multiples
        self.pub_x_field = pub_x_field
        self.pub_y_field = pub_y_field
        self.sig = harmonic_sigs[hex_value]

    def score_all_positions(self):
        """Score all 64 positions for this hex value.

        Returns: (64,) array of resonance scores (higher = more likely).
        """
        scores_ec = np.zeros(64)
        scores_harmonic = np.zeros(64)
        scores_entropy = np.zeros(64)
        scores_proximity = np.zeros(64)
        scores_xor = np.zeros(64)

        for p in range(64):
            contribution = self.hex_multiples[(p, self.h)]

            if contribution == INFINITY:
                remainder = self.target_pub
            else:
                remainder = point_subtract(self.target_pub, contribution)

            rx = remainder.x()
            ry = remainder.y()

            # Oracle 1: EC remainder x-coordinate bit balance
            # Correct remainder should be a "valid" partial sum
            # Score by how "balanced" the x-coordinate bits are
            rx_bits = bin(rx).count("1")
            scores_ec[p] = -abs(rx_bits - 128)  # closer to 128 = more balanced

            # Oracle 2: Harmonic resonance
            # Inner product of this hex value's 4-coeff signature
            # with the corresponding 4 coefficients of the public key field
            # at position p
            start_bit = p * 4
            end_bit = min(start_bit + 4, 256)
            if end_bit > start_bit:
                pub_x_nibble = self.pub_x_field[start_bit:end_bit]
                resonance = np.dot(self.sig[:len(pub_x_nibble)], pub_x_nibble)
                scores_harmonic[p] = resonance

            # Oracle 3: Remainder entropy
            # Lower entropy in remainder x-coord might indicate structure
            rx_hex = f"{rx:064x}"
            nibble_counts = np.array([rx_hex.count(c) for c in HEX_CHARS])
            nibble_freq = nibble_counts / 64.0
            entropy = -np.sum(nibble_freq[nibble_freq > 0] *
                              np.log2(nibble_freq[nibble_freq > 0]))
            scores_entropy[p] = -entropy  # lower entropy = higher score

            # Oracle 4: Proximity to generator multiples
            # Distance between remainder and nearest small multiple of G
            min_dist = float("inf")
            for k in range(1, 17):
                ref = G * k
                dx = abs(rx - ref.x())
                dy = abs(ry - ref.y())
                dist = dx + dy  # Manhattan on field elements
                if dist < min_dist:
                    min_dist = dist
            scores_proximity[p] = -min_dist if min_dist < float("inf") else 0

            # Oracle 5: XOR pattern
            # XOR remainder x with target pub x, count aligned nibbles
            target_x = self.target_pub.x()
            xor_val = rx ^ target_x
            xor_hex = f"{xor_val:064x}"
            zero_nibbles = xor_hex.count("0")
            scores_xor[p] = zero_nibbles

        return {
            "ec_balance": scores_ec,
            "harmonic": scores_harmonic,
            "entropy": scores_entropy,
            "proximity": scores_proximity,
            "xor_pattern": scores_xor,
        }


def collapse(score_matrices, actual_hex_digits):
    """Collapse the superposition: for each position, pick the hex value
    with the highest score across all 16 sandboxes.

    score_matrices: dict[oracle_name -> (16, 64) matrix]
    actual_hex_digits: list of 64 ints (ground truth)
    """
    results = {}
    for oracle_name, matrix in score_matrices.items():
        picks = np.argmax(matrix, axis=0)  # (64,) -- best hex per position
        correct = sum(p == a for p, a in zip(picks, actual_hex_digits))
        results[oracle_name] = {
            "picks": picks,
            "correct": correct,
            "accuracy": correct / 64,
        }
    return results


def ensemble_collapse(score_matrices, actual_hex_digits):
    """Ensemble: normalize each oracle's scores and sum them."""
    combined = np.zeros((16, 64))
    for name, matrix in score_matrices.items():
        # Normalize per position (column-wise)
        for p in range(64):
            col = matrix[:, p]
            col_range = col.max() - col.min()
            if col_range > 0:
                combined[:, p] += (col - col.min()) / col_range
            else:
                combined[:, p] += 1.0 / 16  # uniform

    picks = np.argmax(combined, axis=0)
    correct = sum(p == a for p, a in zip(picks, actual_hex_digits))
    return picks, correct


def bits_correct(predicted_hex, actual_hex):
    """Count how many of the 256 bits match."""
    pred_key = "".join(HEX_CHARS[h] for h in predicted_hex)
    actual_key = "".join(HEX_CHARS[h] for h in actual_hex)
    pred_bits = f"{int(pred_key, 16):0256b}"
    actual_bits = f"{int(actual_key, 16):0256b}"
    return sum(a == b for a, b in zip(pred_bits, actual_bits))


def main():
    print()
    print("=" * 70)
    print("  SUPERPOSITION COLLAPSE ENGINE")
    print("  16 Sandboxes x 64 Positions -- Parallel Resonance Search")
    print("=" * 70)

    # Target key (user's key from previous session)
    target_hex = "06d88f2148757a251dd0ea0e6c4584e159a60cfd3f7217c7b0b111adec0efbca"
    target_key = KeyInput(target_hex)
    target_int = target_key.as_int
    actual_hex_digits = [int(c, 16) for c in target_hex]

    print(f"\n  Target address: {target_hex[:16]}...{target_hex[-16:]}")
    print(f"  (We only use the PUBLIC KEY -- private key is the target)")

    # Derive public key via EC multiply
    print("\n  Deriving public key from private key (this is public info)...")
    t0 = time.time()
    target_pub = G * target_int
    pub_x = target_pub.x()
    pub_y = target_pub.y()
    print(f"  Public key derived in {time.time()-t0:.1f}s")
    print(f"  pub_x: {pub_x:064x}"[:30] + "...")
    print(f"  pub_y: {pub_y:064x}"[:30] + "...")

    # Precompute EC position bases
    position_bases = precompute_position_bases()
    hex_multiples = precompute_hex_multiples(position_bases)

    # Build harmonic representations of the public key
    print("\n  Building harmonic fields for public key...")
    pub_x_key = KeyInput(int(f"{pub_x:064x}"[:64], 16))
    pub_y_key = KeyInput(int(f"{pub_y:064x}"[:64], 16))
    pub_x_bits = np.array(pub_x_key.as_bits, dtype=np.float64)
    pub_y_bits = np.array(pub_y_key.as_bits, dtype=np.float64)
    # Map to +/-1 coefficients
    pub_x_field = 2.0 * pub_x_bits - 1.0
    pub_y_field = 2.0 * pub_y_bits - 1.0

    harmonic_sigs = build_harmonic_signatures()

    # Print the 16 signatures
    print("\n  16 Harmonic Signatures (the 16 sounds):")
    for h in range(16):
        sig_str = "".join("+" if s > 0 else "-" for s in harmonic_sigs[h])
        print(f"    {HEX_CHARS[h]}: [{sig_str}]")

    # ================================================================
    # LAUNCH 16 SANDBOXES
    # ================================================================
    print("\n" + "=" * 70)
    print("  LAUNCHING 16 SANDBOXES (isolated, parallel)")
    print("=" * 70)

    sandboxes = []
    for h in range(16):
        sb = Sandbox(h, target_pub, position_bases, hex_multiples,
                     pub_x_field, pub_y_field, harmonic_sigs)
        sandboxes.append(sb)

    # Each sandbox scores all 64 positions
    print("\n  Sandboxes blaring all 16 sounds across 64 positions...")
    t0 = time.time()

    # Collect scores: dict[oracle_name -> (16, 64) matrix]
    oracle_names = ["ec_balance", "harmonic", "entropy", "proximity", "xor_pattern"]
    score_matrices = {name: np.zeros((16, 64)) for name in oracle_names}

    for h in range(16):
        t1 = time.time()
        scores = sandboxes[h].score_all_positions()
        elapsed = time.time() - t1
        print(f"    Sandbox '{HEX_CHARS[h]}' complete ({elapsed:.1f}s)")

        for name in oracle_names:
            score_matrices[name][h, :] = scores[name]

    total_time = time.time() - t0
    print(f"\n  All 16 sandboxes complete in {total_time:.1f}s")
    print(f"  Total oracle evaluations: {16 * 64 * 5} = 16 x 64 x 5 oracles")

    # ================================================================
    # COLLAPSE THE SUPERPOSITION
    # ================================================================
    print("\n" + "=" * 70)
    print("  COLLAPSING SUPERPOSITION")
    print("=" * 70)

    results = collapse(score_matrices, actual_hex_digits)

    print(f"\n  {'Oracle':20s}  {'Hex Correct':>12s}  {'Bits Correct':>13s}  {'vs Random':>10s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*13}  {'-'*10}")

    for name in oracle_names:
        r = results[name]
        hex_c = r["correct"]
        bit_c = bits_correct(r["picks"], actual_hex_digits)
        expected_hex = 4  # 64/16
        expected_bits = 128  # 256/2
        hex_vs = f"{hex_c - expected_hex:+d}"
        print(f"  {name:20s}  {hex_c:5d}/64      {bit_c:5d}/256     {hex_vs:>10s}")

    # Ensemble (all oracles combined)
    ensemble_picks, ensemble_hex = ensemble_collapse(score_matrices, actual_hex_digits)
    ensemble_bits = bits_correct(ensemble_picks, actual_hex_digits)
    print(f"  {'ENSEMBLE':20s}  {ensemble_hex:5d}/64      {ensemble_bits:5d}/256     {ensemble_hex - 4:+d}")

    # ================================================================
    # VISUAL COLLAPSE
    # ================================================================
    print("\n" + "=" * 70)
    print("  POSITION-BY-POSITION COLLAPSE (Ensemble)")
    print("=" * 70)

    print(f"\n  Pos  Actual  Predicted  Confidence  Match")
    print(f"  ---  ------  ---------  ----------  -----")

    # Compute confidence: margin between best and second-best
    combined = np.zeros((16, 64))
    for name, matrix in score_matrices.items():
        for p in range(64):
            col = matrix[:, p]
            col_range = col.max() - col.min()
            if col_range > 0:
                combined[:, p] += (col - col.min()) / col_range

    for p in range(64):
        col = combined[:, p]
        sorted_scores = np.sort(col)[::-1]
        best_h = np.argmax(col)
        confidence = sorted_scores[0] - sorted_scores[1]  # margin
        actual_h = actual_hex_digits[p]
        match = "YES" if best_h == actual_h else "   "
        print(f"  {p:3d}    {HEX_CHARS[actual_h]}       {HEX_CHARS[best_h]}        {confidence:6.3f}      {match}")

    # ================================================================
    # KEY COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("  ASSEMBLED KEY vs ACTUAL")
    print("=" * 70)

    predicted_key = "".join(HEX_CHARS[h] for h in ensemble_picks)
    print(f"\n  Actual:    {target_hex}")
    print(f"  Predicted: {predicted_key}")

    match_str = ""
    for a, p in zip(target_hex, predicted_key):
        match_str += "^" if a == p else " "
    print(f"  Matches:   {match_str}")
    print(f"\n  Hex positions correct: {ensemble_hex}/64 ({ensemble_hex/64*100:.1f}%)")
    print(f"  Bits correct:          {ensemble_bits}/256 ({ensemble_bits/256*100:.1f}%)")
    print(f"  Expected by chance:    4/64 hex (6.25%), 128/256 bits (50%)")

    # ================================================================
    # CONFIDENCE ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  CONFIDENCE vs CORRECTNESS")
    print("=" * 70)

    confidences = []
    corrects = []
    for p in range(64):
        col = combined[:, p]
        sorted_scores = np.sort(col)[::-1]
        best_h = np.argmax(col)
        confidence = sorted_scores[0] - sorted_scores[1]
        confidences.append(confidence)
        corrects.append(best_h == actual_hex_digits[p])

    confidences = np.array(confidences)
    corrects = np.array(corrects)

    # Bin by confidence quartiles
    quartiles = np.percentile(confidences, [25, 50, 75])
    bins = [
        ("Low (Q1)", confidences <= quartiles[0]),
        ("Med-Low (Q2)", (confidences > quartiles[0]) & (confidences <= quartiles[1])),
        ("Med-High (Q3)", (confidences > quartiles[1]) & (confidences <= quartiles[2])),
        ("High (Q4)", confidences > quartiles[2]),
    ]

    print(f"\n  {'Confidence':15s}  {'N':>4s}  {'Correct':>8s}  {'Accuracy':>9s}")
    print(f"  {'-'*15}  {'-'*4}  {'-'*8}  {'-'*9}")
    for label, mask in bins:
        n = mask.sum()
        c = corrects[mask].sum()
        acc = c / n if n > 0 else 0
        print(f"  {label:15s}  {n:4d}  {c:8d}  {acc:8.1%}")

    # ================================================================
    # THE WALL
    # ================================================================
    print("\n" + "=" * 70)
    print("  WHY THIS HAPPENS")
    print("=" * 70)
    print("""
  The 16 sandboxes each "blare their sound" independently.
  But the resonance target (public key) has NO per-position structure.

  The math:  K = h_0 * 16^63 * G  +  h_1 * 16^62 * G  +  ...  +  h_63 * G

  Subtracting one wrong hex value at one position gives a VALID EC point.
  Subtracting the RIGHT hex value also gives a valid EC point.
  There is no detectable difference -- no "click" when the right sound plays.

  In a real lock, each tumbler gives tactile feedback (the "click").
  EC multiplication gives ZERO per-position feedback.
  You must get ALL 64 positions right simultaneously to verify.

  This is the Discrete Logarithm Problem.
  Grover's Algorithm (quantum) solves it in O(2^128) steps.
  Classical sandboxes: O(16^64) = O(2^256) -- back to brute force.
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
