"""Blind Key Recovery Test.

The honest test: can the harmonic system recover a key it was NEVER given?

Setup:
  1. "Wallet side" generates a random wallet (private key + address)
  2. "Cracker side" receives ONLY the wallet address -- NOT the private key
  3. Cracker tries every technique we have to recover the key
  4. Compare cracker's best guess vs the real private key

This is the real test of whether harmonics can crack crypto.
"""

import sys
import time

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from experiment.crypto_utils import generate_wallet, address_features
from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.core.rip_engine import RipEngine
from quantum_cracker.core.harmonic_compiler import HarmonicCompiler
from quantum_cracker.utils.constants import NUM_THREADS
from quantum_cracker.utils.math_helpers import uniform_sphere_points
from quantum_cracker.utils.types import SimulationConfig

GRID_SIZE = 78


def score_bits(extracted, target):
    matches = sum(a == b for a, b in zip(extracted, target))
    return matches, matches / 256


# =====================================================================
# WALLET SIDE -- generates the secret key (cracker cannot see this)
# =====================================================================

def wallet_side():
    """Generate wallet. Return full wallet dict (secret until reveal)."""
    print("=" * 70)
    print("  WALLET SIDE: Generating secret wallet...")
    print("=" * 70)
    wallet = generate_wallet()
    print(f"  ETH address: {wallet['eth_address']}")
    print(f"  BTC address: {wallet['btc_address']}")
    print(f"  Private key: {'*' * 64}  (HIDDEN)")
    print()
    return wallet


# =====================================================================
# CRACKER SIDE -- receives ONLY the address, tries to find the key
# =====================================================================

def cracker_side(eth_address, btc_address):
    """Try to recover the private key from ONLY the wallet address.

    This is what we'd need to do to actually crack a wallet.
    The cracker has NO access to the private key.
    """
    print("=" * 70)
    print("  CRACKER SIDE: Attempting key recovery from address only")
    print("=" * 70)
    print(f"  Input: ETH address = {eth_address}")
    print(f"  Input: BTC address = {btc_address}")
    print()

    results = {}

    # ------------------------------------------------------------------
    # Attempt 1: Feed the ADDRESS into the harmonic system as if it
    # were a key, and try to read back... something
    # ------------------------------------------------------------------
    print("  [Attempt 1] Feed address bytes into harmonic system...")
    addr_bytes = bytes.fromhex(eth_address.replace("0x", ""))
    # ETH address is 20 bytes (160 bits), not 256 bits
    # Pad to 32 bytes
    padded = addr_bytes + b"\x00" * 12
    try:
        addr_key = KeyInput(padded)
        grid = SphericalVoxelGrid(size=GRID_SIZE)
        grid.initialize_from_key(addr_key)

        # SH readback
        r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
        shell = grid.amplitude[r_mid, :, :]
        theta = np.linspace(0, np.pi, GRID_SIZE)
        phi = np.linspace(0, 2 * np.pi, GRID_SIZE)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        weight = np.sin(theta_grid) * dtheta * dphi

        coefficients = np.zeros(NUM_THREADS, dtype=np.float64)
        bit_idx = 0
        degree = 0
        while bit_idx < NUM_THREADS:
            for m in range(-degree, degree + 1):
                if bit_idx >= NUM_THREADS:
                    break
                ylm = sph_harm_y(degree, m, theta_grid, phi_grid).real
                coefficients[bit_idx] = np.sum(shell * ylm * weight)
                bit_idx += 1
            degree += 1

        addr_bits = [1 if c > 0 else 0 for c in coefficients]
        addr_hex = format(int("".join(str(b) for b in addr_bits), 2), "064x")
        print(f"    Extracted key from address encoding: {addr_hex}")
        print(f"    (This is the address read back, NOT the private key)")
        results["address_encoding"] = addr_bits
    except Exception as e:
        print(f"    Failed: {e}")
        results["address_encoding"] = [0] * 256

    # ------------------------------------------------------------------
    # Attempt 2: Thread z-flip on address-as-key
    # ------------------------------------------------------------------
    print("\n  [Attempt 2] Thread z-flip on address encoding...")
    try:
        base_points = uniform_sphere_points(NUM_THREADS)
        engine = RipEngine()
        engine.initialize_from_key(addr_key)
        actual_dirs = engine.directions.copy()

        zflip_bits = []
        for i in range(NUM_THREADS):
            bz = base_points[i, 2]
            az = actual_dirs[i, 2]
            if bz > 0:
                bit = 0 if az > 0 else 1
            elif bz < 0:
                bit = 0 if az < 0 else 1
            else:
                bit = 0
            zflip_bits.append(bit)

        zflip_hex = format(int("".join(str(b) for b in zflip_bits), 2), "064x")
        print(f"    Z-flip extracted: {zflip_hex}")
        print(f"    (This reads back the ADDRESS bits, not the private key)")
        results["zflip_from_address"] = zflip_bits
    except Exception as e:
        print(f"    Failed: {e}")
        results["zflip_from_address"] = [0] * 256

    # ------------------------------------------------------------------
    # Attempt 3: Random guess (baseline)
    # ------------------------------------------------------------------
    print("\n  [Attempt 3] Random guess (256 coin flips)...")
    random_bits = [int(b) for b in np.random.randint(0, 2, 256)]
    random_hex = format(int("".join(str(b) for b in random_bits), 2), "064x")
    print(f"    Random key: {random_hex}")
    results["random_guess"] = random_bits

    # ------------------------------------------------------------------
    # Attempt 4: The "cheat" -- what our 100% channel actually does
    # This requires the private key, which the cracker does NOT have
    # ------------------------------------------------------------------
    print("\n  [Attempt 4] Thread z-flip (requires private key)...")
    print(f"    ERROR: Cannot run -- private key not available to cracker")
    print(f"    This is WHY the 100% channel is not a real crack.")
    results["cheat_unavailable"] = True

    return results


# =====================================================================
# REVEAL + COMPARISON
# =====================================================================

def reveal_and_compare(wallet, cracker_results):
    """Compare cracker's guesses against the real private key."""
    print()
    print("=" * 70)
    print("  REVEAL: Comparing cracker's guesses to real private key")
    print("=" * 70)

    real_key = KeyInput(wallet["private_key_hex"])
    real_bits = real_key.as_bits
    print(f"\n  Real private key: {wallet['private_key_hex']}")
    print()

    # Score each attempt
    attempts = [
        ("Address encoding -> SH readback", cracker_results["address_encoding"]),
        ("Address encoding -> Z-flip", cracker_results["zflip_from_address"]),
        ("Random guess", cracker_results["random_guess"]),
    ]

    for label, guess_bits in attempts:
        matches, rate = score_bits(guess_bits, real_bits)
        guess_hex = format(int("".join(str(b) for b in guess_bits), 2), "064x")
        print(f"  {label}:")
        print(f"    Guess: {guess_hex}")
        print(f"    Match: {rate:.4f} ({matches}/256 bits)")
        expected = "~50% (random)" if abs(rate - 0.5) < 0.1 else f"{rate:.1%}"
        print(f"    vs random baseline: {expected}")
        print()

    # Now show what the "100% channel" would do IF it had the key
    print("  " + "-" * 66)
    print("  FOR COMPARISON: What the '100%' channel does (WITH the key):")
    print("  " + "-" * 66)

    base_points = uniform_sphere_points(NUM_THREADS)
    engine = RipEngine()
    engine.initialize_from_key(real_key)  # <-- THIS is the step we can't do blind
    actual_dirs = engine.directions.copy()

    cheat_bits = []
    for i in range(NUM_THREADS):
        bz = base_points[i, 2]
        az = actual_dirs[i, 2]
        if bz > 0:
            bit = 0 if az > 0 else 1
        elif bz < 0:
            bit = 0 if az < 0 else 1
        else:
            bit = 0
        cheat_bits.append(bit)

    matches, rate = score_bits(cheat_bits, real_bits)
    print(f"    Match: {rate:.4f} ({matches}/256 bits)")
    print(f"    But this REQUIRED feeding the private key into the engine first.")
    print(f"    initialize_from_key(private_key) --> read directions --> 100%")
    print(f"    Without that first step, we get ~50% (random).")


def main():
    print()
    print("#" * 70)
    print("#  BLIND KEY RECOVERY TEST")
    print("#  Can harmonics crack a key it was never given?")
    print("#" * 70)
    print()

    t0 = time.time()

    # Step 1: Wallet side generates secret
    wallet = wallet_side()

    # Step 2: Cracker side gets ONLY the address
    cracker_results = cracker_side(wallet["eth_address"], wallet["btc_address"])

    # Step 3: Reveal and compare
    reveal_and_compare(wallet, cracker_results)

    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  Without the private key, all attempts score ~50% (random).")
    print(f"  The '100% channel' only works because it reads the key from")
    print(f"  engine memory AFTER you feed it in. It's not cracking -- it's")
    print(f"  reading back what you wrote.")
    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
