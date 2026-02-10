"""Elliptic Curve Point Cloud Analysis.

Instead of a sphere, build a point cloud on secp256k1 starting from
the generator point G. Compute multiples of G (the fundamental
"harmonic" of the curve) and analyze the structure.

The cloud: G, 2G, 3G, ..., N*G -- these are the "harmonics" of the
elliptic curve. Every private key k selects a point k*G from this cloud.

Approach:
1. Build cloud of k*G for k = 1..num_points
2. Treat (x, y) coordinates as signal
3. Apply Fourier/harmonic analysis to the cloud
4. Given a target public key point, try to locate it in the
   harmonic spectrum to recover k
"""

import sys
import time

import numpy as np
from ecdsa import SECP256k1, SigningKey, ellipticcurve

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from experiment.crypto_utils import generate_wallet

# secp256k1 parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
CURVE = SECP256k1.curve
G = ellipticcurve.PointJacobi(CURVE, Gx, Gy, 1, N)


def bits_from_int(value, num_bits=256):
    return [int(b) for b in f"{value:0{num_bits}b}"]


def score_bits(a, b):
    matches = sum(x == y for x, y in zip(a, b))
    return matches, matches / len(a)


def build_cloud(num_points):
    """Build point cloud: k*G for k = 1..num_points.

    Returns arrays of (x, y) coordinates normalized to [0, 1].
    """
    xs = np.zeros(num_points)
    ys = np.zeros(num_points)

    point = G
    for i in range(num_points):
        xs[i] = float(int(point.x())) / float(P)
        ys[i] = float(int(point.y())) / float(P)
        point = point + G  # next multiple

    return xs, ys


def main():
    print()
    print("#" * 70)
    print("#  ELLIPTIC CURVE POINT CLOUD")
    print("#  Cloud of k*G starting from generator G on secp256k1")
    print("#" * 70)

    # Generate target wallet
    wallet = generate_wallet()
    priv_hex = wallet["private_key_hex"]
    priv_int = int(priv_hex, 16)
    priv_bits = bits_from_int(priv_int)
    pub_x = int(wallet["public_key_x"], 16)
    pub_y = int(wallet["public_key_y"], 16)

    print(f"\n  Target public key x: {wallet['public_key_x'][:32]}...")
    print(f"  Target public key y: {wallet['public_key_y'][:32]}...")
    print(f"  Private key: {'*' * 64}")

    # ================================================================
    # Build the cloud
    # ================================================================
    cloud_size = 10000
    print(f"\n  Building point cloud: {cloud_size} multiples of G...")
    t0 = time.time()
    xs, ys = build_cloud(cloud_size)
    elapsed = time.time() - t0
    print(f"  Cloud built in {elapsed:.1f}s")

    # ================================================================
    # Test 1: Is the target point in our cloud?
    # ================================================================
    print()
    print("=" * 70)
    print("TEST 1: Is the target public key in our cloud?")
    print("=" * 70)

    target_x_norm = float(pub_x) / float(P)
    target_y_norm = float(pub_y) / float(P)

    # Find closest point
    distances = np.sqrt((xs - target_x_norm)**2 + (ys - target_y_norm)**2)
    closest_idx = np.argmin(distances)
    closest_dist = distances[closest_idx]

    print(f"  Cloud: k*G for k = 1..{cloud_size}")
    print(f"  Target: {priv_int} * G")
    print(f"  Closest cloud point: k={closest_idx + 1}, distance={closest_dist:.6f}")

    if closest_dist < 1e-10:
        print(f"  FOUND: private key = {closest_idx + 1}")
    else:
        print(f"  NOT FOUND: private key is astronomically larger than our cloud")
        print(f"  (k is a 256-bit number, we only checked k=1..{cloud_size})")

    # ================================================================
    # Test 2: Fourier analysis of the cloud
    # ================================================================
    print()
    print("=" * 70)
    print("TEST 2: Fourier Analysis of G-Cloud")
    print("  Treat x-coordinates of k*G as a signal, look for resonance")
    print("=" * 70)

    # FFT of x-coordinates
    x_centered = xs - np.mean(xs)
    fft_x = np.fft.fft(x_centered)
    power = np.abs(fft_x[:cloud_size//2]) ** 2

    # Spectral flatness (1.0 = white noise, 0.0 = pure tone)
    geometric_mean = np.exp(np.mean(np.log(power[1:] + 1e-30)))
    arithmetic_mean = np.mean(power[1:])
    spectral_flatness = geometric_mean / arithmetic_mean

    print(f"\n  Spectral flatness: {spectral_flatness:.6f}")
    print(f"  (1.0 = white noise, 0.0 = pure tone)")

    if spectral_flatness > 0.9:
        print(f"  Cloud x-coordinates are effectively random -- no harmonic structure")
    elif spectral_flatness > 0.5:
        print(f"  Some spectral coloring but mostly noise-like")
    else:
        print(f"  Strong spectral peaks -- possible structure to exploit!")

    top_modes = np.argsort(power[1:])[::-1][:10] + 1
    print(f"\n  Top 10 spectral modes:")
    for k in top_modes:
        print(f"    Mode {k:5d}: power = {power[k]:.2e}")

    # ================================================================
    # Test 3: Can we locate the target by projecting onto cloud harmonics?
    # ================================================================
    print()
    print("=" * 70)
    print("TEST 3: Project Target onto Cloud Harmonics")
    print("  Use cloud as basis, project target point, read off coefficients")
    print("=" * 70)

    # Build a "basis" from the first 256 cloud points
    # (like using G, 2G, ..., 256G as basis vectors)
    basis_x = xs[:256]  # normalized x-coords of G, 2G, ..., 256G
    basis_y = ys[:256]

    # Project target onto each basis vector
    # Inner product: target dot basis_k (treating (x,y) as 2D vectors)
    projections = np.zeros(256)
    for i in range(256):
        projections[i] = target_x_norm * basis_x[i] + target_y_norm * basis_y[i]

    # Convert projections to bits
    proj_bits = [1 if p > np.median(projections) else 0 for p in projections]
    matches, rate = score_bits(proj_bits, priv_bits)

    print(f"\n  Projection onto 256 G-multiples:")
    print(f"    Match with private key bits: {matches}/256 ({rate:.4f})")
    print(f"    Expected (random): ~128/256 (0.5000)")

    # ================================================================
    # Test 4: Harmonic resonance on the cloud
    # ================================================================
    print()
    print("=" * 70)
    print("TEST 4: Resonance Sweep on G-Cloud")
    print("  Apply sin/cos at different frequencies to cloud coordinates")
    print("  Check if any frequency reveals the private key")
    print("=" * 70)

    best_rate = 0.0
    best_freq = 0

    for freq in range(1, 201):
        # Evaluate harmonic at target point
        signal_x = np.sin(freq * 2 * np.pi * target_x_norm)
        signal_y = np.cos(freq * 2 * np.pi * target_y_norm)

        # Evaluate same harmonic at cloud points
        cloud_signal_x = np.sin(freq * 2 * np.pi * xs[:256])
        cloud_signal_y = np.cos(freq * 2 * np.pi * ys[:256])

        # Compare target signal with each cloud signal
        diff_x = np.abs(cloud_signal_x - signal_x)
        diff_y = np.abs(cloud_signal_y - signal_y)
        total_diff = diff_x + diff_y

        # Bits: 1 if target is "close" to this cloud point at this frequency
        threshold = np.median(total_diff)
        freq_bits = [1 if d < threshold else 0 for d in total_diff]

        matches, rate = score_bits(freq_bits, priv_bits)
        if rate > best_rate:
            best_rate = rate
            best_freq = freq

    print(f"\n  Swept 200 frequencies")
    print(f"  Best frequency: {best_freq} (match: {best_rate:.4f}, {int(best_rate*256)}/256)")
    print(f"  Expected (random): ~0.5000")

    # ================================================================
    # Test 5: The fundamental problem
    # ================================================================
    print()
    print("=" * 70)
    print("TEST 5: The Fundamental Problem Demonstrated")
    print("=" * 70)

    print(f"\n  Private key space: 2^256 = ~1.16 x 10^77 possible keys")
    print(f"  Our cloud size:   {cloud_size} points")
    print(f"  Coverage:          {cloud_size / 2**256:.2e} of the space")
    print(f"")
    print(f"  To check every possible key at {cloud_size}/sec:")
    years = (2**256 / cloud_size) / (365.25 * 24 * 3600)
    print(f"  Time needed: ~10^{np.log10(years):.0f} years")
    print(f"  Age of universe: ~10^10 years")
    print(f"")
    print(f"  The cloud approach cannot scale. The group order of secp256k1")
    print(f"  is ~2^256. To find k from k*G, you'd need to search the entire")
    print(f"  group -- which is the discrete logarithm problem (ECDLP).")
    print(f"  No amount of harmonic analysis changes the search space.")

    # Reveal
    print()
    print("=" * 70)
    print("  REVEAL")
    print("=" * 70)
    print(f"  Private key: {priv_hex}")
    print(f"  This key is point #{priv_int} in the G-cloud.")
    print(f"  Our cloud only covered points 1 through {cloud_size}.")
    print("=" * 70)


if __name__ == "__main__":
    main()
