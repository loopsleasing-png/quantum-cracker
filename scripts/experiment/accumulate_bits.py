"""Bit Accumulation Test.

Run many iterations with different frequencies/strengths/steps.
Track which bits are correct each time. Keep only the bits that
are consistently right across runs.

If there's signal: certain bits stay correct across all runs.
If it's random: surviving bits decay exponentially toward zero.
  - After 1 run: ~128 bits correct
  - After 2 runs: ~64 bits correct in BOTH
  - After 3 runs: ~32 bits correct in ALL THREE
  - After N runs: ~256 * (0.5)^N bits survive
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


def sh_readback_at_config(key_input, freq, steps, strength):
    """Run resonance with given config, extract bits via SH readback."""
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
    return bits


def main():
    print()
    print("#" * 70)
    print("#  BIT ACCUMULATION TEST")
    print("#  Record correct bits each run, keep only survivors")
    print("#" * 70)

    # Generate secret wallet
    wallet = generate_wallet()
    real_key = KeyInput(wallet["private_key_hex"])
    real_bits = np.array(real_key.as_bits)

    print(f"\n  ETH address: {wallet['eth_address']}")
    print(f"  Private key: {'*' * 64}  (hidden)")

    # Pad address to 32 bytes
    addr_bytes = bytes.fromhex(wallet["eth_address"].replace("0x", ""))
    addr_padded = addr_bytes + b"\x00" * 12
    addr_key = KeyInput(addr_padded)

    # Generate many different configurations
    configs = []
    for freq in [1, 5, 10, 16, 20, 30, 40, 50, 60, 70, 78, 80, 90, 100, 110, 120, 130, 140, 150, 156]:
        for strength in [0.01, 0.05, 0.10, 0.15]:
            for steps in [100, 200, 500]:
                configs.append((freq, steps, strength))

    print(f"  Configurations to test: {len(configs)}")
    print()

    # Track per-bit correctness across ALL runs
    # survived[i] = True if bit i has been correct in every run so far
    survived = np.ones(256, dtype=bool)  # all start as "survived"
    per_bit_correct_count = np.zeros(256, dtype=int)
    total_runs = 0

    # Track the decay
    decay_log = []  # (run_number, surviving_bits, this_run_matches, expected_random)

    t0 = time.time()

    print(f"  {'Run':>4s}  {'Freq':>4s}  {'Steps':>5s}  {'Str':>5s}  "
          f"{'This Run':>8s}  {'Survivors':>9s}  {'Expected':>8s}  {'Majority':>8s}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}")

    for ci, (freq, steps, strength) in enumerate(configs):
        bits = sh_readback_at_config(addr_key, freq, steps, strength)
        bits_arr = np.array(bits)

        # Which bits are correct this run?
        correct_this_run = (bits_arr == real_bits)
        match_count = int(np.sum(correct_this_run))

        # Update per-bit count
        per_bit_correct_count += correct_this_run.astype(int)
        total_runs += 1

        # Update survivors: must be correct in ALL runs
        survived = survived & correct_this_run
        surviving_count = int(np.sum(survived))

        # Expected survivors if random: 256 * (0.5)^N
        expected_random = 256 * (0.5 ** total_runs)

        # Majority vote so far: for each bit, was it correct in >50% of runs?
        majority_correct = np.sum(per_bit_correct_count > total_runs / 2)

        decay_log.append((total_runs, surviving_count, match_count, expected_random))

        if ci < 20 or ci % 10 == 0 or surviving_count == 0:
            print(f"  {total_runs:4d}  {freq:4d}  {steps:5d}  {strength:5.2f}  "
                  f"{match_count:5d}/256  {surviving_count:6d}/256  "
                  f"{expected_random:8.1f}  {int(majority_correct):5d}/256")

        if surviving_count == 0 and total_runs > 10:
            # Keep going for majority vote stats but stop printing survivors
            pass

    total_time = time.time() - t0

    # ================================================================
    # FINAL REPORT
    # ================================================================
    print()
    print("=" * 70)
    print("  BIT ACCUMULATION RESULTS")
    print("=" * 70)

    print(f"\n  Total runs: {total_runs}")
    print(f"  Final survivors (correct in ALL {total_runs} runs): {int(np.sum(survived))}/256")

    # When did survivors hit zero?
    zero_run = next((r for r, s, _, _ in decay_log if s == 0), None)
    if zero_run:
        print(f"  Survivors hit zero at run: {zero_run}")
        expected_zero = int(np.ceil(np.log2(256)))  # log2(256) = 8
        print(f"  Expected (random) to hit zero at run: ~{expected_zero}")

    # Majority vote
    majority_bits = (per_bit_correct_count > total_runs / 2).astype(int)
    majority_matches = int(np.sum(majority_bits == real_bits))
    print(f"\n  Majority vote ({total_runs} runs): {majority_matches}/256 ({majority_matches/256:.4f})")
    print(f"  Expected (random): ~128/256")

    # Per-bit success rate distribution
    per_bit_rate = per_bit_correct_count / total_runs
    print(f"\n  Per-bit success rate across {total_runs} runs:")
    print(f"    Mean:   {np.mean(per_bit_rate):.4f}  (random = 0.5000)")
    print(f"    Std:    {np.std(per_bit_rate):.4f}")
    print(f"    Min:    {np.min(per_bit_rate):.4f}")
    print(f"    Max:    {np.max(per_bit_rate):.4f}")

    # Any bits significantly above chance?
    # With 240 runs, a bit at 50% has std = sqrt(0.25/240) = 0.032
    std_per_bit = np.sqrt(0.25 / total_runs)
    threshold_3sigma = 0.5 + 3 * std_per_bit
    above_3sigma = np.sum(per_bit_rate > threshold_3sigma)
    print(f"\n  Bits with success rate > 3-sigma ({threshold_3sigma:.4f}): {above_3sigma}")
    expected_above = 256 * 0.0013  # ~0.13% above 3 sigma by chance
    print(f"  Expected by chance: ~{expected_above:.1f}")

    # Decay curve
    print(f"\n  SURVIVOR DECAY CURVE:")
    print(f"  {'Run':>4s}  {'Survivors':>9s}  {'Expected (random)':>17s}")
    milestones = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 150, 200, 240]
    for run, surv, _, exp in decay_log:
        if run in milestones:
            print(f"  {run:4d}  {surv:6d}/256   {exp:14.1f}/256")

    # ================================================================
    # REVEAL
    # ================================================================
    print()
    print("=" * 70)
    print("  REVEAL")
    print("=" * 70)
    print(f"  Real key:      {wallet['private_key_hex']}")
    maj_hex = format(int("".join(str(b) for b in majority_bits), 2), "064x")
    print(f"  Majority vote: {maj_hex}")
    print(f"  Match:         {majority_matches}/256")
    print()

    if majority_matches > 140:
        print(f"  POSSIBLE SIGNAL: {majority_matches}/256 > 140")
    else:
        print(f"  NO SIGNAL: {majority_matches}/256 -- random noise.")
        print(f"  Accumulating more runs cannot reveal a key that isn't there.")
        print(f"  The address is a one-way hash of the key. No amount of")
        print(f"  harmonic analysis on the address can reverse the hash.")

    print(f"\n  Runtime: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
