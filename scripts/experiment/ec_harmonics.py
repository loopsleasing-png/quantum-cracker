"""Elliptic Curve Harmonic Analysis.

Replace the sphere with secp256k1: y^2 = x^3 + 7

Approach:
1. Parametrize the real-valued curve y^2 = x^3 + 7
2. Define Fourier-like basis functions along the curve
3. Encode private key bits as coefficients
4. Perform EC scalar multiplication (over the actual finite field)
5. See if the public key point reveals anything about the coefficients

Also: direct analysis of EC scalar multiplication structure.
Does k*G have any detectable harmonic pattern as a function of k?
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from ecdsa import SECP256k1, SigningKey, VerifyingKey, ellipticcurve
from experiment.crypto_utils import generate_wallet

# secp256k1 parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def ec_multiply(k):
    """Compute k*G on secp256k1, return (x, y) as integers."""
    curve = SECP256k1.curve
    G = ellipticcurve.PointJacobi(curve, Gx, Gy, 1, N)
    point = k * G
    return (int(point.x()), int(point.y()))


def bits_from_int(value, num_bits=256):
    """Extract bits from an integer (MSB first)."""
    return [int(b) for b in f"{value:0{num_bits}b}"]


def score_bits(extracted, target):
    matches = sum(a == b for a, b in zip(extracted, target))
    return matches, matches / len(target)


def test_1_scalar_mult_patterns():
    """Test: does k*G show harmonic patterns as k varies?

    If we compute k*G for k = 1, 2, 3, ..., N, do the x-coordinates
    of the resulting points show any spectral structure?

    We test a small sample and look at the distribution.
    """
    print("=" * 70)
    print("TEST 1: Scalar Multiplication Patterns")
    print("  Computing k*G for k = 1..1000, analyzing x-coordinate distribution")
    print("=" * 70)

    x_coords = []
    y_coords = []
    for k in range(1, 1001):
        x, y = ec_multiply(k)
        x_coords.append(x)
        y_coords.append(y)

    x_arr = np.array(x_coords, dtype=np.float64)
    y_arr = np.array(y_coords, dtype=np.float64)

    # Normalize to [0, 1] for analysis
    x_norm = x_arr / float(P)
    y_norm = y_arr / float(P)

    # Check autocorrelation of x-coordinates
    # If there's harmonic structure, autocorrelation will show periodicity
    x_centered = x_norm - np.mean(x_norm)
    autocorr = np.correlate(x_centered, x_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # positive lags only
    autocorr /= autocorr[0]  # normalize

    print(f"\n  x-coordinate statistics (k=1..1000):")
    print(f"    Mean (normalized): {np.mean(x_norm):.6f}  (uniform = 0.500)")
    print(f"    Std:               {np.std(x_norm):.6f}  (uniform = 0.289)")
    print(f"    Autocorrelation lag-1: {autocorr[1]:.6f}  (random = ~0.000)")
    print(f"    Autocorrelation lag-2: {autocorr[2]:.6f}")
    print(f"    Autocorrelation lag-5: {autocorr[5]:.6f}")
    print(f"    Autocorrelation lag-10: {autocorr[10]:.6f}")

    # FFT of x-coordinates
    fft = np.fft.fft(x_centered)
    power = np.abs(fft[:500]) ** 2
    peak_freq = np.argmax(power[1:]) + 1  # skip DC
    print(f"\n    FFT peak frequency: {peak_freq} (out of 500)")
    print(f"    Peak power ratio:   {power[peak_freq] / np.mean(power[1:]):.2f}x mean")

    if power[peak_freq] / np.mean(power[1:]) > 5:
        print(f"    POSSIBLE SPECTRAL STRUCTURE DETECTED")
    else:
        print(f"    No spectral structure -- looks random (as expected)")

    return x_norm, y_norm


def test_2_pubkey_bit_correlation():
    """Test: do bits of the public key x-coordinate correlate with private key bits?

    For 100 random private keys, compute pubkey = k*G.
    Check if any bits of pubkey_x consistently predict bits of k.
    """
    print()
    print("=" * 70)
    print("TEST 2: Public Key Bit Correlation with Private Key")
    print("  100 random keys: do pubkey_x bits predict privkey bits?")
    print("=" * 70)

    n_keys = 100
    priv_bits_matrix = np.zeros((n_keys, 256), dtype=int)
    pubx_bits_matrix = np.zeros((n_keys, 256), dtype=int)
    puby_bits_matrix = np.zeros((n_keys, 256), dtype=int)

    for i in range(n_keys):
        sk = SigningKey.generate(curve=SECP256k1)
        priv_int = int.from_bytes(sk.to_string(), "big")
        vk = sk.get_verifying_key()
        pub_bytes = vk.to_string()
        pub_x = int.from_bytes(pub_bytes[:32], "big")
        pub_y = int.from_bytes(pub_bytes[32:], "big")

        priv_bits_matrix[i] = bits_from_int(priv_int)
        pubx_bits_matrix[i] = bits_from_int(pub_x)
        puby_bits_matrix[i] = bits_from_int(pub_y)

    # For each (privkey_bit, pubkey_bit) pair, compute correlation
    # This is a 256x256 correlation matrix
    print(f"\n  Computing 256x256 bit correlation matrix...")

    max_corr = 0.0
    max_corr_pos = (0, 0)
    corr_above_threshold = 0

    # Sample: check every 4th pair to save time
    for pi in range(0, 256, 1):
        for qi in range(0, 256, 4):
            priv_col = priv_bits_matrix[:, pi].astype(float)
            pubx_col = pubx_bits_matrix[:, qi].astype(float)

            # Skip constant columns
            if np.std(priv_col) < 0.01 or np.std(pubx_col) < 0.01:
                continue

            corr = abs(np.corrcoef(priv_col, pubx_col)[0, 1])
            if corr > max_corr:
                max_corr = corr
                max_corr_pos = (pi, qi)
            if corr > 0.3:
                corr_above_threshold += 1

    print(f"    Max |correlation|:     {max_corr:.4f} (privkey bit {max_corr_pos[0]}, pubkey_x bit {max_corr_pos[1]})")
    print(f"    Pairs with |r| > 0.3: {corr_above_threshold}")
    print(f"    Expected by chance:   ~{int(256*64*0.003)} (from {256*64} pairs tested)")

    if max_corr > 0.5:
        print(f"    POSSIBLE CORRELATION DETECTED")
    else:
        print(f"    No meaningful correlations -- EC multiplication is a strong mixer")


def test_3_ec_curve_fourier():
    """Test: Fourier analysis on the elliptic curve itself.

    Parametrize y^2 = x^3 + 7 over the reals.
    Sample points along the curve, compute Fourier coefficients.
    See if the curve has natural resonant frequencies.
    """
    print()
    print("=" * 70)
    print("TEST 3: Fourier Analysis on y^2 = x^3 + 7 (Real Curve)")
    print("=" * 70)

    # y^2 = x^3 + 7, so y = sqrt(x^3 + 7) for x >= -7^(1/3) ~ -1.913
    # Parametrize by x from the cusp outward
    x_min = -(7 ** (1/3))  # where y = 0
    x_max = 100.0

    n_points = 2048
    x_vals = np.linspace(x_min + 0.001, x_max, n_points)
    y_vals = np.sqrt(x_vals**3 + 7)

    # Arc length parametrization
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    s_norm = s / s[-1]  # normalize to [0, 1]

    # Fourier analysis of x(s) and y(s)
    x_centered = x_vals - np.mean(x_vals)
    y_centered = y_vals - np.mean(y_vals)

    fft_x = np.fft.fft(x_centered)
    fft_y = np.fft.fft(y_centered)
    power_x = np.abs(fft_x[:n_points//2]) ** 2
    power_y = np.abs(fft_y[:n_points//2]) ** 2

    print(f"\n  Curve parametrization: {n_points} points, x in [{x_min:.3f}, {x_max}]")
    print(f"  Arc length: {s[-1]:.2f}")

    # Top spectral peaks
    top_x = np.argsort(power_x[1:])[::-1][:5] + 1
    top_y = np.argsort(power_y[1:])[::-1][:5] + 1

    print(f"\n  Top 5 Fourier modes of x(s):")
    for k in top_x:
        print(f"    Mode {k:4d}: power = {power_x[k]:.2e}")

    print(f"\n  Top 5 Fourier modes of y(s):")
    for k in top_y:
        print(f"    Mode {k:4d}: power = {power_y[k]:.2e}")

    # The curve is dominated by low-frequency modes (it's smooth and monotonically growing)
    # No sharp resonances expected
    ratio = power_x[top_x[0]] / np.mean(power_x[1:])
    print(f"\n  Spectral concentration: top mode is {ratio:.1f}x mean")
    print(f"  This just reflects that the curve is smooth -- not a useful resonance")


def test_4_blind_ec_crack():
    """Test: use EC curve structure to try to crack a key.

    Given public key (x, y), try to recover private key k.
    Use the public key coordinates as input to harmonic analysis.
    """
    print()
    print("=" * 70)
    print("TEST 4: Blind EC Crack Attempt")
    print("  Given public key point (x, y), try to recover private key")
    print("=" * 70)

    # Generate a wallet
    wallet = generate_wallet()
    priv_hex = wallet["private_key_hex"]
    pub_x_hex = wallet["public_key_x"]
    pub_y_hex = wallet["public_key_y"]
    priv_int = int(priv_hex, 16)
    pub_x_int = int(pub_x_hex, 16)
    pub_y_int = int(pub_y_hex, 16)

    priv_bits = bits_from_int(priv_int)

    print(f"\n  Public key x: {pub_x_hex[:32]}...")
    print(f"  Public key y: {pub_y_hex[:32]}...")
    print(f"  Private key:  {'*' * 64}")

    # Attempt A: use pub_x bits directly as private key guess
    pubx_bits = bits_from_int(pub_x_int)
    matches_x, rate_x = score_bits(pubx_bits, priv_bits)
    print(f"\n  Attempt A: pub_x bits as private key guess")
    print(f"    Match: {matches_x}/256 ({rate_x:.4f})")

    # Attempt B: use pub_y bits
    puby_bits = bits_from_int(pub_y_int)
    matches_y, rate_y = score_bits(puby_bits, priv_bits)
    print(f"\n  Attempt B: pub_y bits as private key guess")
    print(f"    Match: {matches_y}/256 ({rate_y:.4f})")

    # Attempt C: XOR pub_x and pub_y
    xor_int = pub_x_int ^ pub_y_int
    xor_bits = bits_from_int(xor_int)
    matches_xor, rate_xor = score_bits(xor_bits, priv_bits)
    print(f"\n  Attempt C: pub_x XOR pub_y")
    print(f"    Match: {matches_xor}/256 ({rate_xor:.4f})")

    # Attempt D: Fourier analysis of pub_x bit sequence
    # Treat the 256 bits as a signal, take FFT, use phase as key guess
    pubx_signal = np.array(pubx_bits, dtype=float) * 2 - 1  # map to +/-1
    fft = np.fft.fft(pubx_signal)
    phase = np.angle(fft)
    phase_bits = [1 if p > 0 else 0 for p in phase[:256]]
    matches_fft, rate_fft = score_bits(phase_bits, priv_bits)
    print(f"\n  Attempt D: FFT phase of pub_x bits")
    print(f"    Match: {matches_fft}/256 ({rate_fft:.4f})")

    # Attempt E: multiply pub_x by curve parameter (x^3 + 7), take bits
    # This is using the curve equation directly
    rhs = (pub_x_int ** 3 + 7) % P
    rhs_bits = bits_from_int(rhs)
    matches_rhs, rate_rhs = score_bits(rhs_bits, priv_bits)
    print(f"\n  Attempt E: x^3 + 7 mod p bits")
    print(f"    Match: {matches_rhs}/256 ({rate_rhs:.4f})")

    # Attempt F: sqrt of (x^3 + 7) should give y -- verify, then use remainder
    # y^2 = x^3 + 7, so y^2 - (x^3 + 7) = 0 mod p
    verify = (pub_y_int ** 2 - pub_x_int ** 3 - 7) % P
    print(f"\n  Curve verification: y^2 - x^3 - 7 mod p = {verify}  (should be 0)")

    # Attempt G: generator point analysis
    # k*G = (pub_x, pub_y). If we compute (pub_x - Gx) mod p, does it relate to k?
    delta_x = (pub_x_int - Gx) % P
    delta_bits = bits_from_int(delta_x)
    matches_delta, rate_delta = score_bits(delta_bits, priv_bits)
    print(f"\n  Attempt G: (pub_x - Gx) mod p bits")
    print(f"    Match: {matches_delta}/256 ({rate_delta:.4f})")

    # Summary
    print(f"\n  SUMMARY:")
    attempts = [
        ("pub_x direct", rate_x),
        ("pub_y direct", rate_y),
        ("pub_x XOR pub_y", rate_xor),
        ("FFT phase of pub_x", rate_fft),
        ("x^3 + 7 mod p", rate_rhs),
        ("(pub_x - Gx) mod p", rate_delta),
        ("Random baseline", 0.5),
    ]
    for label, rate in attempts:
        signal = "SIGNAL" if rate > 0.55 else "random"
        print(f"    {label:25s}: {rate:.4f}  [{signal}]")

    # Reveal
    print(f"\n  Private key: {priv_hex}")
    best_rate = max(rate_x, rate_y, rate_xor, rate_fft, rate_rhs, rate_delta)
    print(f"  Best attempt: {best_rate:.4f} ({int(best_rate*256)}/256)")


def test_5_nearby_key_analysis():
    """Test: does k+1 produce a public key near k's public key?

    If nearby private keys produce nearby public keys, there might
    be a gradient we could follow. If not, EC mult is a strong mixer.
    """
    print()
    print("=" * 70)
    print("TEST 5: Nearby Key Analysis")
    print("  Does k+1 produce a nearby public key to k?")
    print("=" * 70)

    sk = SigningKey.generate(curve=SECP256k1)
    k = int.from_bytes(sk.to_string(), "big")

    x0, y0 = ec_multiply(k)
    x1, y1 = ec_multiply(k + 1)
    x2, y2 = ec_multiply(k + 2)
    x10, y10 = ec_multiply(k + 10)
    x100, y100 = ec_multiply(k + 100)

    def bit_dist(a, b):
        """Hamming distance between bit representations."""
        bits_a = bits_from_int(a)
        bits_b = bits_from_int(b)
        return sum(x != y for x, y in zip(bits_a, bits_b))

    print(f"\n  Private key k: {k % (10**20)}... (last 20 digits)")
    print(f"\n  Bit distance between public key x-coordinates:")
    print(f"    k vs k+1:   {bit_dist(x0, x1):3d}/256 bits differ  (random = ~128)")
    print(f"    k vs k+2:   {bit_dist(x0, x2):3d}/256 bits differ")
    print(f"    k vs k+10:  {bit_dist(x0, x10):3d}/256 bits differ")
    print(f"    k vs k+100: {bit_dist(x0, x100):3d}/256 bits differ")

    print(f"\n  Bit distance between public key y-coordinates:")
    print(f"    k vs k+1:   {bit_dist(y0, y1):3d}/256 bits differ")
    print(f"    k vs k+2:   {bit_dist(y0, y2):3d}/256 bits differ")
    print(f"    k vs k+10:  {bit_dist(y0, y10):3d}/256 bits differ")
    print(f"    k vs k+100: {bit_dist(y0, y100):3d}/256 bits differ")

    if bit_dist(x0, x1) < 100:
        print(f"\n  GRADIENT DETECTED: nearby keys produce nearby pubkeys")
    else:
        print(f"\n  No gradient: changing k by 1 scrambles all 256 bits of the public key.")
        print(f"  EC scalar multiplication is an extremely strong mixer.")
        print(f"  There is no smooth landscape to hill-climb on.")


def main():
    print()
    print("#" * 70)
    print("#  ELLIPTIC CURVE HARMONIC ANALYSIS")
    print("#  Exploring y^2 = x^3 + 7 (secp256k1)")
    print("#" * 70)

    t0 = time.time()

    test_1_scalar_mult_patterns()
    test_2_pubkey_bit_correlation()
    test_3_ec_curve_fourier()
    test_4_blind_ec_crack()
    test_5_nearby_key_analysis()

    total = time.time() - t0

    print()
    print("=" * 70)
    print("  OVERALL CONCLUSION")
    print("=" * 70)
    print(f"  1. Scalar multiplication on secp256k1 produces pseudo-random output")
    print(f"  2. No bit-level correlation between private key and public key")
    print(f"  3. The real curve has smooth Fourier modes but they don't help")
    print(f"     (the crypto operates over a finite field, not the reals)")
    print(f"  4. All blind crack attempts score ~50% (random)")
    print(f"  5. Changing k by 1 flips ~128/256 bits of the public key")
    print(f"     -- no gradient to follow, no smooth landscape to search")
    print(f"\n  The elliptic curve discrete logarithm problem (ECDLP) is hard")
    print(f"  because EC multiplication is an algebraic one-way function that")
    print(f"  completely destroys any exploitable structure. Harmonic analysis")
    print(f"  on the curve surface cannot reverse this.")
    print(f"\n  Runtime: {total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
