"""Phase 2: Differential Harmonic Analysis (DHA).

Novel approach: flip ONE private key bit, measure how the PUBLIC KEY's
harmonic profile changes. If the delta is consistent across different
base keys, that's a per-bit signal exploitable for cracking.

For each bit position:
  1. Generate 50 random base keys
  2. Flip bit i in each
  3. Compute both public keys via EC multiply
  4. Feed pub_x of each through harmonic pipeline
  5. Measure delta in SH coefficients
  6. Check if deltas are consistent (high cosine similarity)
"""

import csv
import secrets
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import Point

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.utils.math_helpers import build_qr_sh_basis

G = SECP256k1.generator
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()
ORDER = SECP256k1.order

GRID_SIZE = 20  # small for speed
N_MODES = 256
N_KEYS_PER_BIT = 50
BIT_POSITIONS = list(range(0, 256, 4))  # every 4th bit = 64 positions


def pub_key_to_harmonic_coeffs(pub_x_int):
    """Convert a public key x-coordinate to SH coefficients."""
    # Clamp to 256 bits
    pub_x_hex = f"{pub_x_int % (2**256):064x}"
    key = KeyInput(pub_x_hex)
    bits = np.array(key.as_bits, dtype=np.float64)
    coeffs = 2.0 * bits - 1.0  # map to +/-1

    basis = build_qr_sh_basis(GRID_SIZE, N_MODES)
    n_points = GRID_SIZE * GRID_SIZE

    if n_points < N_MODES:
        # Can't do full projection, use raw dot product
        return coeffs
    else:
        # Project onto QR basis and read back
        field = basis @ coeffs
        # Read back coefficients
        readback = basis.T @ field
        return readback


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main():
    print()
    print("=" * 70)
    print("  PHASE 2: DIFFERENTIAL HARMONIC ANALYSIS (DHA)")
    print(f"  {len(BIT_POSITIONS)} bit positions x {N_KEYS_PER_BIT} keys = {len(BIT_POSITIONS)*N_KEYS_PER_BIT} EC multiplies")
    print("=" * 70)

    # Pre-warm the QR basis cache
    print(f"\n  Pre-building QR basis ({GRID_SIZE}x{GRID_SIZE}, {N_MODES} modes)...")
    build_qr_sh_basis(GRID_SIZE, N_MODES)

    results = []
    t_total = time.time()

    for idx, bit_pos in enumerate(BIT_POSITIONS):
        t0 = time.time()
        deltas = []

        for _ in range(N_KEYS_PER_BIT):
            # Random base key
            k_base = int.from_bytes(secrets.token_bytes(32), "big") % ORDER
            if k_base == 0:
                k_base = 1

            # Flip bit at position bit_pos
            # bit_pos 0 = MSB (bit 255 in integer), bit_pos 255 = LSB (bit 0)
            flip_mask = 1 << (255 - bit_pos)
            k_flip = k_base ^ flip_mask

            # Compute both public keys
            K_base = G * k_base
            K_flip = G * k_flip

            # Get harmonic coefficients of pub_x
            coeffs_base = pub_key_to_harmonic_coeffs(K_base.x())
            coeffs_flip = pub_key_to_harmonic_coeffs(K_flip.x())

            # Delta
            delta = coeffs_flip - coeffs_base
            deltas.append(delta)

        deltas = np.array(deltas)  # (N_KEYS_PER_BIT, N_MODES)

        # Analyze consistency of deltas
        # 1. Mean cosine similarity between all pairs
        n = len(deltas)
        cos_sims = []
        for i in range(min(n, 20)):  # sample pairs for speed
            for j in range(i + 1, min(n, 20)):
                cs = cosine_sim(deltas[i], deltas[j])
                cos_sims.append(cs)

        mean_cos = np.mean(cos_sims) if cos_sims else 0.0
        std_cos = np.std(cos_sims) if cos_sims else 0.0

        # 2. Signal-to-noise ratio per coefficient
        mean_delta = np.mean(deltas, axis=0)
        std_delta = np.std(deltas, axis=0)
        snr = np.abs(mean_delta) / (std_delta + 1e-10)
        max_snr = float(np.max(snr))
        mean_snr = float(np.mean(snr))

        # 3. Is this bit "crackable"?
        is_crackable = mean_cos > 0.3

        elapsed = time.time() - t0
        status = "SIGNAL" if is_crackable else "noise"
        print(f"  Bit {bit_pos:3d}  cos_sim={mean_cos:+.4f}  max_snr={max_snr:.3f}  "
              f"mean_snr={mean_snr:.3f}  [{status}]  ({elapsed:.1f}s)")

        results.append({
            "bit_position": bit_pos,
            "mean_cosine_similarity": mean_cos,
            "std_cosine_similarity": std_cos,
            "max_snr": max_snr,
            "mean_snr": mean_snr,
            "is_crackable": is_crackable,
        })

    total_time = time.time() - t_total
    print(f"\n  Total time: {total_time:.1f}s")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    cos_sims_all = [r["mean_cosine_similarity"] for r in results]
    snrs_all = [r["max_snr"] for r in results]
    crackable = [r for r in results if r["is_crackable"]]

    print(f"\n  Cosine similarity distribution across {len(BIT_POSITIONS)} bit positions:")
    print(f"    Mean: {np.mean(cos_sims_all):.4f}")
    print(f"    Std:  {np.std(cos_sims_all):.4f}")
    print(f"    Min:  {np.min(cos_sims_all):.4f}")
    print(f"    Max:  {np.max(cos_sims_all):.4f}")

    print(f"\n  Max SNR distribution:")
    print(f"    Mean: {np.mean(snrs_all):.4f}")
    print(f"    Max:  {np.max(snrs_all):.4f}")

    if crackable:
        print(f"\n  *** {len(crackable)} CRACKABLE BIT(S) DETECTED (cos_sim > 0.3) ***")
        for r in crackable:
            print(f"    Bit {r['bit_position']}: cos_sim={r['mean_cosine_similarity']:.4f}, "
                  f"max_snr={r['max_snr']:.4f}")
    else:
        print(f"\n  No crackable bits. EC multiply fully randomizes per-bit deltas.")
        print(f"  Expected: cosine similarities near 0 (random delta directions).")

    # ================================================================
    # WRITE CSV
    # ================================================================
    csv_path = "/Users/kjm/Desktop/phase2_dha_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Results written to {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
