"""Parity-Driven ECDLP Experiment -- Scaling to 128 bits.

Tests whether PDQM parity dynamics provide any advantage over standard
MCMC for the elliptic curve discrete logarithm problem, from 8-bit
toy curves up to 128-bit curves.

Methods compared:
  1. Random guessing (baseline)
  2. Standard Metropolis MCMC (single-spin flips only)
  3. PDQM parity Glauber (pair hopping + parity suppression)
  4. PDQM parity oracle (annealing + parity-weighted voting) [N <= 20 only]

Usage:
    python scripts/experiment/parity_ecdlp.py
    python scripts/experiment/parity_ecdlp.py --bits 8,16,32,64,128
    python scripts/experiment/parity_ecdlp.py --bits 8,10,12 --n-trials 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

sys.path.insert(0, "src")

from quantum_cracker.parity.dynamics import ParityDynamics
from quantum_cracker.parity.ec_constraints import SmallEC, make_curve
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.oracle import ParityOracle
from quantum_cracker.parity.types import AnnealSchedule, ParityConfig


@dataclass
class TrialResult:
    curve_p: int
    n_bits: int
    method: str
    true_key: int
    extracted_key: int
    bit_match_rate: float
    key_recovered: bool
    energy_final: float
    elapsed_sec: float
    trial_idx: int
    sweeps_or_traj: int


def bit_match(k1: int, k2: int, n: int) -> float:
    xor = k1 ^ k2
    matching = n - bin(xor).count("1")
    return matching / n


def run_random_baseline(true_key: int, n_bits: int, rng: np.random.Generator) -> tuple[int, float]:
    if n_bits > 63:
        import secrets
        guess = secrets.randbelow(1 << n_bits)
    else:
        guess = int(rng.integers(0, 1 << n_bits))
    return guess, bit_match(guess, true_key, n_bits)


def run_standard_mcmc(
    h: ParityHamiltonian, true_key: int, n_sweeps: int, temp: float,
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    n = h.n_spins
    sigma0 = rng.choice([-1, 1], size=n).astype(np.int8)
    dyn = ParityDynamics(h, h.config)
    snaps = dyn.evolve_standard_mcmc(
        sigma0, n_sweeps=n_sweeps, temperature=temp,
        target_key=true_key, log_interval=n_sweeps, rng=rng,
    )
    final_key = ParityHamiltonian.spins_to_key(snaps[-1].spins)
    return final_key, bit_match(final_key, true_key, n), snaps[-1].energy


def run_parity_glauber(
    h: ParityHamiltonian, config: ParityConfig, true_key: int,
    n_sweeps: int, temp: float, rng: np.random.Generator,
) -> tuple[int, float, float]:
    n = h.n_spins
    sigma0 = rng.choice([-1, 1], size=n).astype(np.int8)
    dyn = ParityDynamics(h, config)
    snaps = dyn.evolve_glauber(
        sigma0, n_sweeps=n_sweeps, temperature=temp,
        target_key=true_key, log_interval=n_sweeps, rng=rng,
    )
    final_key = ParityHamiltonian.spins_to_key(snaps[-1].spins)
    return final_key, bit_match(final_key, true_key, n), snaps[-1].energy


def run_parity_oracle(
    h: ParityHamiltonian, config: ParityConfig, true_key: int,
    n_traj: int, anneal_steps: int, rng: np.random.Generator,
) -> tuple[int, float, float]:
    oracle = ParityOracle(h, config)
    schedule = AnnealSchedule(n_steps=anneal_steps, beta_initial=0.1, beta_final=20.0)
    result = oracle.measure(n_trajectories=n_traj, schedule=schedule, target_key=true_key, rng=rng)
    mr = oracle.bit_match_rate(result, true_key)
    ek = oracle.extract_key(result)
    return ek, mr, result.best_energy


def main():
    parser = argparse.ArgumentParser(description="Parity ECDLP Scaling Experiment")
    parser.add_argument("--bits", type=str, default="8,10,12,16,20,24,32,48,64,80,96,112,128",
                        help="Comma-separated target bit sizes")
    parser.add_argument("--n-trials", type=int, default=5, help="Trials per curve per method")
    parser.add_argument("--base-sweeps", type=int, default=200, help="Base MCMC sweeps (scales with N)")
    parser.add_argument("--oracle-traj", type=int, default=100, help="Oracle trajectories (N<=20 only)")
    parser.add_argument("--oracle-steps", type=int, default=500, help="Oracle anneal steps")
    parser.add_argument("--temperature", type=float, default=0.2, help="MCMC temperature")
    parser.add_argument("--delta-e", type=float, default=2.0, help="Parity energy gap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    bit_sizes = [int(b.strip()) for b in args.bits.split(",")]
    rng = np.random.default_rng(args.seed)
    all_results: list[TrialResult] = []

    print("=" * 78)
    print(" PARITY-DRIVEN ECDLP SCALING EXPERIMENT")
    print("=" * 78)
    print(f"  Bit sizes:    {bit_sizes}")
    print(f"  Trials/method: {args.n_trials}")
    print(f"  Base sweeps:   {args.base_sweeps}")
    print(f"  Oracle:        {args.oracle_traj} traj x {args.oracle_steps} steps (N<=20)")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Delta_E:       {args.delta_e}")
    print(f"  Seed:          {args.seed}")
    print("=" * 78)
    print()

    for target_bits in bit_sizes:
        curve = make_curve(target_bits)
        G = curve.generator
        n_bits = curve.key_bit_length()

        # Scale sweeps with N: more bits need more exploration
        n_sweeps = max(args.base_sweeps, args.base_sweeps * n_bits // 8)

        can_oracle = (n_bits <= 20)
        methods = ["random", "standard_mcmc", "parity_glauber"]
        if can_oracle:
            methods.append("parity_oracle")

        print(f"--- {target_bits}-bit target -> {n_bits}-bit curve (p={str(curve.p)[:30]}) ---")
        print(f"    sweeps={n_sweeps}, oracle={'YES' if can_oracle else 'NO (N>{20})'}")

        config = ParityConfig(
            n_spins=n_bits,
            delta_e=args.delta_e,
            j_coupling=0.1,
            t1_base=0.05,
            t2=1.0,
            temperature=args.temperature,
            mode="exact" if n_bits <= 20 else "ising",
        )

        method_rates: dict[str, list[float]] = {m: [] for m in methods}

        for trial in range(args.n_trials):
            k, P = curve.random_keypair(rng)
            h = ParityHamiltonian.from_ec_dlp(curve, G, P, config)

            # 1. Random
            t0 = time.time()
            ek_r, mr_r = run_random_baseline(k, n_bits, rng)
            dt_r = time.time() - t0
            method_rates["random"].append(mr_r)
            all_results.append(TrialResult(
                curve_p=curve.p, n_bits=n_bits, method="random",
                true_key=k, extracted_key=ek_r, bit_match_rate=mr_r,
                key_recovered=(ek_r == k), energy_final=0.0,
                elapsed_sec=dt_r, trial_idx=trial, sweeps_or_traj=0,
            ))

            # 2. Standard MCMC
            t0 = time.time()
            ek_s, mr_s, e_s = run_standard_mcmc(h, k, n_sweeps, args.temperature, rng)
            dt_s = time.time() - t0
            method_rates["standard_mcmc"].append(mr_s)
            all_results.append(TrialResult(
                curve_p=curve.p, n_bits=n_bits, method="standard_mcmc",
                true_key=k, extracted_key=ek_s, bit_match_rate=mr_s,
                key_recovered=(ek_s == k), energy_final=e_s,
                elapsed_sec=dt_s, trial_idx=trial, sweeps_or_traj=n_sweeps,
            ))

            # 3. Parity Glauber
            t0 = time.time()
            ek_pg, mr_pg, e_pg = run_parity_glauber(h, config, k, n_sweeps, args.temperature, rng)
            dt_pg = time.time() - t0
            method_rates["parity_glauber"].append(mr_pg)
            all_results.append(TrialResult(
                curve_p=curve.p, n_bits=n_bits, method="parity_glauber",
                true_key=k, extracted_key=ek_pg, bit_match_rate=mr_pg,
                key_recovered=(ek_pg == k), energy_final=e_pg,
                elapsed_sec=dt_pg, trial_idx=trial, sweeps_or_traj=n_sweeps,
            ))

            # 4. Oracle (small N only)
            if can_oracle:
                t0 = time.time()
                ek_po, mr_po, e_po = run_parity_oracle(
                    h, config, k, args.oracle_traj, args.oracle_steps, rng
                )
                dt_po = time.time() - t0
                method_rates["parity_oracle"].append(mr_po)
                all_results.append(TrialResult(
                    curve_p=curve.p, n_bits=n_bits, method="parity_oracle",
                    true_key=k, extracted_key=ek_po, bit_match_rate=mr_po,
                    key_recovered=(ek_po == k), energy_final=e_po,
                    elapsed_sec=dt_po, trial_idx=trial,
                    sweeps_or_traj=args.oracle_traj,
                ))

            sys.stdout.write(f"\r    trial {trial+1}/{args.n_trials}")
            sys.stdout.flush()

        print()
        # Summary for this curve
        print(f"    {'Method':<20} {'Mean Match':>12} {'Std':>8} {'Keys':>8} {'Avg Time':>10}")
        print(f"    {'-'*58}")
        for method in methods:
            rates = method_rates[method]
            arr = np.array(rates)
            n_found = sum(
                1 for r in all_results
                if r.n_bits == n_bits and r.method == method and r.key_recovered
            )
            times = [
                r.elapsed_sec for r in all_results
                if r.n_bits == n_bits and r.method == method
            ]
            avg_time = np.mean(times) if times else 0
            print(f"    {method:<20} {arr.mean():>12.4f} {arr.std():>8.4f} {n_found:>5}/{len(rates)} {avg_time:>9.3f}s")
        print()

    # === Grand Summary ===
    print("=" * 78)
    print(" GRAND SUMMARY -- SCALING BEHAVIOR")
    print("=" * 78)
    print()

    # Per-method summary across all sizes
    for method in ["random", "standard_mcmc", "parity_glauber", "parity_oracle"]:
        results_m = [r for r in all_results if r.method == method]
        if not results_m:
            continue
        rates = [r.bit_match_rate for r in results_m]
        n_found = sum(1 for r in results_m if r.key_recovered)
        arr = np.array(rates)
        print(f"  {method:<20}  mean={arr.mean():.4f}  std={arr.std():.4f}  keys={n_found}/{len(rates)}")

    # Scaling table: match rate vs bit size
    print()
    print(f"  {'Bits':>6}  {'Random':>10}  {'Std MCMC':>10}  {'Parity GL':>10}  {'Oracle':>10}")
    print(f"  {'-'*50}")
    for target_bits in bit_sizes:
        row = f"  {target_bits:>6}"
        for method in ["random", "standard_mcmc", "parity_glauber", "parity_oracle"]:
            rates = [r.bit_match_rate for r in all_results
                     if r.method == method and r.n_bits == make_curve(target_bits).key_bit_length()]
            if rates:
                row += f"  {np.mean(rates):>10.4f}"
            else:
                row += f"  {'---':>10}"
        print(row)

    # Statistical comparison: parity glauber vs standard MCMC
    print()
    pg_rates = np.array([r.bit_match_rate for r in all_results if r.method == "parity_glauber"])
    sm_rates = np.array([r.bit_match_rate for r in all_results if r.method == "standard_mcmc"])
    if len(pg_rates) > 0 and len(sm_rates) > 0:
        diff = pg_rates.mean() - sm_rates.mean()
        pooled = np.sqrt(pg_rates.var() / len(pg_rates) + sm_rates.var() / len(sm_rates))
        z = diff / pooled if pooled > 0 else 0
        print(f"  Parity Glauber vs Standard MCMC (all sizes):")
        print(f"    Mean diff: {diff:+.4f}  Z-score: {z:.2f}", end="")
        if z > 1.96:
            print("  -> REJECT null (p < 0.05)")
        elif z > 1.64:
            print("  -> MARGINAL (p < 0.10)")
        else:
            print("  -> FAIL TO REJECT null")

    print("=" * 78)

    # === Export CSV ===
    desktop = os.path.expanduser("~/Desktop")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(desktop, f"parity_ecdlp_scaling_{timestamp}.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_bits", "method", "trial", "bit_match_rate", "key_recovered",
            "energy_final", "elapsed_sec", "sweeps_or_traj",
        ])
        for r in all_results:
            writer.writerow([
                r.n_bits, r.method, r.trial_idx, f"{r.bit_match_rate:.4f}",
                r.key_recovered, f"{r.energy_final:.4f}", f"{r.elapsed_sec:.4f}",
                r.sweeps_or_traj,
            ])

    print(f"\nResults exported to {filepath}")


if __name__ == "__main__":
    main()
