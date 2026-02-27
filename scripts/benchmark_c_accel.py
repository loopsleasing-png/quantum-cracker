#!/usr/bin/env python3
"""Benchmark C accelerator vs pure Python EC evaluator.

Measures the speedup of the C implementation for the critical
inner loop operations: set_state, flip_single, peek_flip_single,
and peek_flip_pair.
"""

from __future__ import annotations

import time
import numpy as np

from quantum_cracker.parity.ec_constraints import (
    ECConstraintEncoder,
    ECEnergyEvaluator,
    SmallEC,
)


def benchmark_evaluator(ev, label: str, n_ops: int = 100_000) -> dict:
    """Benchmark evaluator operations."""
    rng = np.random.default_rng(42)
    n_bits = ev.n_bits

    # Benchmark set_state
    t0 = time.perf_counter()
    for k in range(min(n_ops, 1000)):
        ev.set_state(k % 79)
    t_set = time.perf_counter() - t0

    # Benchmark flip_single
    ev.set_state(5)
    t0 = time.perf_counter()
    for _ in range(n_ops):
        bit = int(rng.integers(0, n_bits))
        ev.flip_single(bit)
    t_flip = time.perf_counter() - t0

    # Benchmark peek_flip_single
    ev.set_state(5)
    t0 = time.perf_counter()
    for _ in range(n_ops):
        bit = int(rng.integers(0, n_bits))
        ev.peek_flip_single(bit)
    t_peek = time.perf_counter() - t0

    # Benchmark peek_flip_pair
    ev.set_state(5)
    t0 = time.perf_counter()
    for _ in range(n_ops):
        a = int(rng.integers(0, n_bits))
        b = int(rng.integers(0, n_bits))
        ev.peek_flip_pair(a, b)
    t_pair = time.perf_counter() - t0

    results = {
        "label": label,
        "set_state": t_set,
        "flip_single": t_flip,
        "peek_flip_single": t_peek,
        "peek_flip_pair": t_pair,
    }
    return results


def main() -> None:
    curve = SmallEC(97, 0, 7)
    G = curve.generator
    key = 5
    P = curve.multiply(G, key)
    assert P is not None
    n_bits = curve.key_bit_length()

    n_ops = 100_000

    # Pure Python evaluator
    py_ev = ECEnergyEvaluator(curve, G, P, n_bits)
    py_results = benchmark_evaluator(py_ev, "Python", n_ops)

    # C evaluator
    try:
        from quantum_cracker.accel._ec_arith import CECEvaluator
        c_ev = CECEvaluator(curve.p, curve.a, curve.b, G[0], G[1], P[0], P[1], n_bits)
        c_results = benchmark_evaluator(c_ev, "C", n_ops)
    except (ImportError, RuntimeError):
        print("C accelerator not available")
        c_results = None

    print(f"\nBenchmark: {n_ops:,} operations on curve y^2=x^3+7 over F_97")
    print(f"{'Operation':<25} {'Python (s)':<14} {'C (s)':<14} {'Speedup':<10}")
    print("-" * 63)

    for key in ["set_state", "flip_single", "peek_flip_single", "peek_flip_pair"]:
        py_t = py_results[key]
        if c_results:
            c_t = c_results[key]
            speedup = py_t / c_t if c_t > 0 else float("inf")
            print(f"{key:<25} {py_t:<14.4f} {c_t:<14.4f} {speedup:<10.1f}x")
        else:
            print(f"{key:<25} {py_t:<14.4f} {'N/A':<14}")


if __name__ == "__main__":
    main()
