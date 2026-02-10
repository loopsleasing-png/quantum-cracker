"""Blind Frequency Sweep Test.

Feed ONLY the wallet address into the harmonic system at many frequencies.
Record which bits match the real private key at each frequency.
Check if any bits consistently emerge across frequencies.

If harmonics can crack crypto, we'd see certain bits lock in
across frequencies (convergence). If it's random, each frequency
gives ~50% with no consistency.
"""

import sys
import time

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from experiment.crypto_utils import generate_wallet
from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.utils.constants import NUM_THREADS

GRID_SIZE = 78
FREQUENCIES = list(range(1, 157))  # 1-156 MHz


def sh_readback_at_freq(key_input, freq, steps=200, strength=0.05):
    """Run resonance at given frequency, extract bits via SH readback."""
    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key_input)

    _, theta_grid, phi_grid = np.meshgrid(
        grid.r_coords, grid.theta_coords, grid.phi_coords, indexing="ij"
    )

    for step in range(steps):
        t = step * 0.01
        vibration = np.sin(freq * phi_grid + t) * np.cos(freq * theta_grid)
        grid.amplitude *= 1.0 + vibration * strength
        grid.energy = np.abs(grid.amplitude) ** 2

    # SH readback on mid-radius shell
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :]

    theta = np.linspace(0, np.pi, GRID_SIZE)
    phi = np.linspace(0, 2 * np.pi, GRID_SIZE)
    tg, pg = np.meshgrid(theta, phi, indexing="ij")
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    weight = np.sin(tg) * dtheta * dphi

    coefficients = np.zeros(NUM_THREADS, dtype=np.float64)
    bit_idx = 0
    degree = 0
    while bit_idx < NUM_THREADS:
        for m in range(-degree, degree + 1):
            if bit_idx >= NUM_THREADS:
                break
            ylm = sph_harm_y(degree, m, tg, pg).real
            coefficients[bit_idx] = np.sum(shell * ylm * weight)
            bit_idx += 1
        degree += 1

    bits = [1 if c > 0 else 0 for c in coefficients]
    return bits, coefficients


def main():
    print()
    print("#" * 70)
    print("#  BLIND FREQUENCY SWEEP -- Address Only, 156 Frequencies")
    print("#" * 70)

    # Generate secret wallet
    wallet = generate_wallet()
    real_key = KeyInput(wallet["private_key_hex"])
    real_bits = real_key.as_bits

    print(f"\n  ETH address: {wallet['eth_address']}")
    print(f"  Private key: {'*' * 64}  (hidden until reveal)")
    print()

    # Pad address to 32 bytes for KeyInput
    addr_bytes = bytes.fromhex(wallet["eth_address"].replace("0x", ""))
    addr_padded = addr_bytes + b"\x00" * 12
    addr_key = KeyInput(addr_padded)

    # Sweep all frequencies
    bit_matrix = np.zeros((len(FREQUENCIES), 256), dtype=int)  # extracted bits
    match_matrix = np.zeros((len(FREQUENCIES), 256), dtype=int)  # 1 if correct
    freq_rates = []

    t0 = time.time()
    print(f"  Sweeping {len(FREQUENCIES)} frequencies...")
    print()

    for fi, freq in enumerate(FREQUENCIES):
        bits, coeffs = sh_readback_at_freq(addr_key, freq)
        bit_matrix[fi] = bits
        matches = [1 if bits[i] == real_bits[i] else 0 for i in range(256)]
        match_matrix[fi] = matches
        match_count = sum(matches)
        rate = match_count / 256
        freq_rates.append(rate)

        elapsed = time.time() - t0
        eta = elapsed / (fi + 1) * (len(FREQUENCIES) - fi - 1)
        if freq % 10 == 0 or freq <= 5:
            print(f"    freq={freq:3d} MHz: {rate:.4f} ({match_count}/256)  "
                  f"[{fi+1}/{len(FREQUENCIES)}] {elapsed:.0f}s, ~{eta:.0f}s left")

    total_time = time.time() - t0

    # ================================================================
    # ANALYSIS
    # ================================================================
    print()
    print("=" * 70)
    print("  RESULTS -- BLIND FREQUENCY SWEEP")
    print("=" * 70)

    # Overall stats
    print(f"\n  Frequencies tested: {len(FREQUENCIES)}")
    print(f"  Mean match rate:    {np.mean(freq_rates):.4f}")
    print(f"  Best match rate:    {max(freq_rates):.4f} (freq={FREQUENCIES[np.argmax(freq_rates)]} MHz)")
    print(f"  Worst match rate:   {min(freq_rates):.4f} (freq={FREQUENCIES[np.argmin(freq_rates)]} MHz)")
    print(f"  Std dev:            {np.std(freq_rates):.4f}")
    print(f"  Expected (random):  0.5000 +/- {1/np.sqrt(256):.4f}")

    # Per-bit consistency: how many frequencies get each bit right?
    per_bit_correct = np.sum(match_matrix, axis=0)  # out of 156

    print(f"\n  PER-BIT CONSISTENCY (across {len(FREQUENCIES)} frequencies):")
    print(f"    Bits correct in ALL frequencies:   {np.sum(per_bit_correct == len(FREQUENCIES))}")
    print(f"    Bits correct in >75% of frequencies: {np.sum(per_bit_correct > len(FREQUENCIES) * 0.75)}")
    print(f"    Bits correct in >60% of frequencies: {np.sum(per_bit_correct > len(FREQUENCIES) * 0.60)}")
    print(f"    Bits correct in ~50% of frequencies: {np.sum((per_bit_correct > len(FREQUENCIES) * 0.40) & (per_bit_correct < len(FREQUENCIES) * 0.60))}")
    print(f"    Bits correct in <25% of frequencies: {np.sum(per_bit_correct < len(FREQUENCIES) * 0.25)}")
    print(f"    Bits correct in ZERO frequencies:  {np.sum(per_bit_correct == 0)}")

    # Expected distribution for random: binomial(156, 0.5)
    expected_mean = len(FREQUENCIES) * 0.5
    expected_std = np.sqrt(len(FREQUENCIES) * 0.25)
    print(f"\n    Expected if random: mean={expected_mean:.1f}, std={expected_std:.1f}")
    print(f"    Actual:             mean={np.mean(per_bit_correct):.1f}, std={np.std(per_bit_correct):.1f}")

    # Majority vote: for each bit, take the majority across all frequencies
    majority_bits = []
    for i in range(256):
        ones = np.sum(bit_matrix[:, i])
        majority_bits.append(1 if ones > len(FREQUENCIES) / 2 else 0)

    majority_matches = sum(majority_bits[i] == real_bits[i] for i in range(256))
    majority_rate = majority_matches / 256

    print(f"\n  MAJORITY VOTE (all {len(FREQUENCIES)} frequencies combined):")
    print(f"    Match: {majority_rate:.4f} ({majority_matches}/256)")
    print(f"    Expected (random): ~0.5000")

    # Confidence-weighted vote
    # (doesn't help if underlying data is random, but let's try)
    print()

    # Top 10 frequencies
    top_freqs = sorted(range(len(FREQUENCIES)), key=lambda i: freq_rates[i], reverse=True)[:10]
    print(f"  TOP 10 FREQUENCIES:")
    for fi in top_freqs:
        print(f"    {FREQUENCIES[fi]:3d} MHz: {freq_rates[fi]:.4f} ({int(freq_rates[fi]*256)}/256)")

    # Majority from top 10 only
    top10_majority = []
    for i in range(256):
        ones = sum(bit_matrix[fi, i] for fi in top_freqs)
        top10_majority.append(1 if ones > 5 else 0)
    top10_matches = sum(top10_majority[i] == real_bits[i] for i in range(256))
    print(f"\n  Majority vote (top 10 frequencies): {top10_matches}/256 ({top10_matches/256:.4f})")

    # ================================================================
    # REVEAL
    # ================================================================
    print()
    print("=" * 70)
    print("  REVEAL")
    print("=" * 70)
    print(f"  Real private key: {wallet['private_key_hex']}")
    majority_hex = format(int("".join(str(b) for b in majority_bits), 2), "064x")
    print(f"  Best guess (vote): {majority_hex}")
    print(f"  Match:             {majority_matches}/256 ({majority_rate:.4f})")
    print()

    if majority_rate > 0.55:
        print(f"  SIGNAL DETECTED: {majority_rate:.1%} > 55% -- worth investigating!")
    else:
        print(f"  NO SIGNAL: {majority_rate:.1%} is within random chance (~50%).")
        print(f"  The address contains no recoverable information about the private key.")
        print(f"  Hash functions (keccak256) destroy all structure.")

    print(f"\n  Runtime: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
