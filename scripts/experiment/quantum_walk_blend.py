"""Quantum Walk Coin Blending -- Finding the Sweet Spot.

The full geometry coins destroyed walk coherence (all zeros).
The Hadamard coin works but ignores geometry.

This experiment: interpolate between them.

  angle[v] = (1 - alpha) * pi/4 + alpha * geometry_angle[v]

  alpha = 0.00 : pure Hadamard (uniform, no geometry)
  alpha = 0.01 : 1% geometry nudge
  alpha = 0.50 : half and half
  alpha = 1.00 : full geometry (what killed coherence before)

If there's a sweet spot where a SMALL amount of geometry improves
the walk beyond plain Hadamard, that's the signal.

Analogy: acoustic impedance matching. Too much geometry = reflection.
Just enough = the wave follows the curve's structure.
"""

import csv
import math
import sys
import time

import numpy as np
from scipy import stats
from scipy.special import sph_harm_y

sys.path.insert(0, "src")


# ================================================================
# SMALL EC (same as quantum_walk_ec.py)
# ================================================================

class SmallEC:
    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b
        self.points = self._enumerate_points()
        self.order = len(self.points)
        self.generator = None
        self._point_to_idx = {pt: i for i, pt in enumerate(self.points)}

    def _enumerate_points(self):
        points = [None]
        p, a, b = self.p, self.a, self.b
        qr = {}
        for y in range(p):
            qr.setdefault((y * y) % p, []).append(y)
        for x in range(p):
            rhs = (x * x * x + a * x + b) % p
            if rhs in qr:
                for y in qr[rhs]:
                    points.append((x, y))
        return points

    def add(self, P, Q):
        if P is None: return Q
        if Q is None: return P
        p = self.p
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and y1 == (p - y2) % p:
            return None
        if P == Q:
            if y1 == 0: return None
            inv = pow(2 * y1, p - 2, p)
            lam = (3 * x1 * x1 + self.a) * inv % p
        else:
            if x1 == x2: return None
            inv = pow((x2 - x1) % p, p - 2, p)
            lam = (y2 - y1) * inv % p
        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def multiply(self, P, k):
        if k == 0 or P is None: return None
        result = None
        addend = P
        while k:
            if k & 1: result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def find_generator(self):
        for pt in self.points[1:]:
            if self.multiply(pt, self.order) is None:
                is_gen = True
                for d in range(2, int(self.order**0.5) + 1):
                    if self.order % d == 0:
                        if self.multiply(pt, self.order // d) is None:
                            is_gen = False
                            break
                if is_gen:
                    self.generator = pt
                    return pt
        self.generator = self.points[1]
        return self.generator


# ================================================================
# VECTORIZED QUANTUM WALK
# ================================================================

def run_blended_walk(N, base_angle, geo_angles, alpha, start, target, max_steps):
    """Run quantum walk with blended coin angles.

    Coin angle at node v = (1 - alpha) * base_angle + alpha * geo_angles[v]
    """
    blended = (1.0 - alpha) * base_angle + alpha * geo_angles
    cos_a = np.cos(blended)
    sin_a = np.sin(blended)

    left = np.zeros(N, dtype=np.complex128)
    right = np.zeros(N, dtype=np.complex128)
    left[start] = 1.0 / np.sqrt(2)
    right[start] = 1.0 / np.sqrt(2)

    max_prob = abs(left[target])**2 + abs(right[target])**2
    max_step = 0
    # Track probability curve at coarse intervals
    curve_points = []

    for step in range(1, max_steps + 1):
        new_left = cos_a * left + sin_a * right
        new_right = -sin_a * left + cos_a * right
        left = np.roll(new_left, -1)
        right = np.roll(new_right, 1)

        p_target = abs(left[target])**2 + abs(right[target])**2
        if p_target > max_prob:
            max_prob = p_target
            max_step = step

        if step % max(1, max_steps // 50) == 0:
            curve_points.append((step, p_target))

    return max_prob, max_step, curve_points


# ================================================================
# GEOMETRY ANGLE GENERATORS
# ================================================================

def angles_coordinate(ec, cycle):
    """Coin angles from (x, y) coordinates."""
    N = len(cycle)
    p = ec.p
    angles = np.zeros(N)
    for i, pt in enumerate(cycle):
        if pt is None:
            angles[i] = np.pi / 4
        else:
            x, y = pt
            angles[i] = np.pi * ((x * 0.618033988 + y * 0.381966) % p) / p
    return angles


def angles_curvature(ec, cycle):
    """Coin angles from curve curvature (dy/dx)."""
    N = len(cycle)
    p = ec.p
    angles = np.zeros(N)
    for i, pt in enumerate(cycle):
        if pt is None:
            angles[i] = np.pi / 4
        else:
            x, y = pt
            if y == 0:
                angles[i] = np.pi / 2
            else:
                dydx_num = (3 * x * x + ec.a) % p
                dydx_den = (2 * y) % p
                ratio = (dydx_num * pow(dydx_den, p - 2, p)) % p
                angles[i] = np.pi * ratio / p
    return angles


def angles_xmod(ec, cycle):
    """Coin angles from x mod small primes."""
    N = len(cycle)
    p = ec.p
    angles = np.zeros(N)
    for i, pt in enumerate(cycle):
        if pt is None:
            angles[i] = np.pi / 4
        else:
            x, y = pt
            # Use x mod 7 as a simple geometric feature
            angles[i] = 2 * np.pi * (x % 7) / 7
    return angles


def angles_hamming(ec, cycle):
    """Coin angles from Hamming weight of x-coordinate."""
    N = len(cycle)
    angles = np.zeros(N)
    for i, pt in enumerate(cycle):
        if pt is None:
            angles[i] = np.pi / 4
        else:
            x, y = pt
            hw = bin(x).count('1')
            angles[i] = np.pi * hw / 16  # normalize assuming <=16 bits
    return angles


def angles_residual(ec, cycle):
    """Coin angles from y^2 - x^3 - ax - b (pre-mod value)."""
    N = len(cycle)
    p = ec.p
    angles = np.zeros(N)
    for i, pt in enumerate(cycle):
        if pt is None:
            angles[i] = np.pi / 4
        else:
            x, y = pt
            raw = y * y - x * x * x - ec.a * x - ec.b
            # This is 0 mod p but has structure before mod
            angles[i] = np.pi * ((raw % (p * p)) / (p * p))
    return angles


def angles_position_in_cycle(ec, cycle):
    """Coin angles from position in the group cycle (distance from identity).

    This encodes the GROUP STRUCTURE directly.
    On a plain cycle this would be i/N * 2pi.
    On EC, the "natural" position IS the discrete log.
    """
    N = len(cycle)
    angles = np.zeros(N)
    for i in range(N):
        angles[i] = 2 * np.pi * i / N
    return angles


def angles_harmonic_sh(ec, cycle, n_harm=8):
    """Coin angles from SH decomposition of point positions on sphere."""
    N = len(cycle)
    p = ec.p
    thetas = np.zeros(N)
    phis = np.zeros(N)
    for i, pt in enumerate(cycle):
        if pt is None:
            thetas[i] = 0; phis[i] = 0
        else:
            x, y = pt
            thetas[i] = np.pi * x / p
            phis[i] = 2 * np.pi * y / p

    sh_vals = np.zeros((N, n_harm))
    idx = 0; l = 0
    while idx < n_harm:
        for m in range(-l, l + 1):
            if idx >= n_harm: break
            sh_vals[:, idx] = sph_harm_y(l, m, thetas, phis).real
            idx += 1
        l += 1

    weights = np.array([1.0 / (k + 1) for k in range(n_harm)])
    angles = np.pi / 4 + 0.5 * np.tanh(sh_vals @ weights)  # centered on pi/4
    return angles


# ================================================================
# MAIN
# ================================================================

def main():
    print()
    print("=" * 78)
    print("  QUANTUM WALK COIN BLENDING -- Finding the Sweet Spot")
    print("  Interpolate: alpha=0 (pure Hadamard) to alpha=1 (full geometry)")
    print("=" * 78)

    # Curves to test
    curve_configs = [
        (211, 0, 7, "p211"),
        (251, 0, 7, "p251"),
        (503, 0, 7, "p503"),
        (1009, 0, 7, "p1009"),
        (2003, 0, 7, "p2003"),
        (4091, 0, 7, "p4091"),
        (8191, 0, 7, "p8191"),
    ]

    # Alpha sweep values -- fine grain near 0 where the sweet spot likely is
    alphas = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
              0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

    # Geometry coin types
    coin_types = [
        ("coordinate", angles_coordinate),
        ("curvature", angles_curvature),
        ("xmod7", angles_xmod),
        ("hamming", angles_hamming),
        ("residual", angles_residual),
        ("cycle_pos", angles_position_in_cycle),
        ("harmonic_sh", angles_harmonic_sh),
    ]

    base_angle = np.pi / 4  # Hadamard angle

    all_results = []

    for p_val, a, b, name in curve_configs:
        print(f"\n  Building {name}: y^2 = x^3 + {a}x + {b} mod {p_val}")
        ec = SmallEC(p_val, a, b)
        gen = ec.find_generator()
        if gen is None:
            print(f"    No generator, skipping")
            continue

        # Build cycle
        cycle = []
        P = None
        for k in range(ec.order):
            cycle.append(P)
            P = ec.add(P, gen)
        N = len(cycle)

        target_k = N // 3 + 7
        target_k = target_k % N
        max_steps = min(N * 4, 5000)

        print(f"    |E|={ec.order}, N={N}, target={target_k}, steps={max_steps}")

        # Baseline: pure Hadamard (alpha=0)
        uniform_angles = np.full(N, base_angle)
        baseline_prob, baseline_step, _ = run_blended_walk(
            N, base_angle, uniform_angles, 0.0, 0, target_k, max_steps)
        print(f"    Hadamard baseline: max_P={baseline_prob:.6f} at step {baseline_step}")

        # Sweep each coin type x alpha
        for coin_name, coin_fn in coin_types:
            geo_angles = coin_fn(ec, cycle)

            best_alpha = 0.0
            best_prob = baseline_prob
            best_step = baseline_step
            alpha_probs = []

            t0 = time.time()
            for alpha in alphas:
                mp, ms, _ = run_blended_walk(
                    N, base_angle, geo_angles, alpha, 0, target_k, max_steps)
                alpha_probs.append(mp)
                if mp > best_prob:
                    best_prob = mp
                    best_alpha = alpha
                    best_step = ms
            elapsed = time.time() - t0

            improvement = best_prob / baseline_prob if baseline_prob > 0 else 0
            marker = " ***" if improvement > 1.1 else ""

            print(f"    {coin_name:14s}: best_alpha={best_alpha:.3f}  "
                  f"max_P={best_prob:.6f} ({improvement:.2f}x baseline)  "
                  f"step={best_step}  ({elapsed:.1f}s){marker}")

            all_results.append({
                "curve": name,
                "N": N,
                "coin": coin_name,
                "baseline_prob": baseline_prob,
                "best_alpha": best_alpha,
                "best_prob": best_prob,
                "improvement": improvement,
                "best_step": best_step,
                "alpha_probs": alpha_probs,
            })

    # ================================================================
    # CROSS-CURVE ANALYSIS
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  CROSS-CURVE ANALYSIS: Which coins and alphas work?")
    print(f"  {'='*74}")

    # Group by coin type
    for coin_name, _ in coin_types:
        coin_results = [r for r in all_results if r["coin"] == coin_name]
        if not coin_results:
            continue

        improvements = [r["improvement"] for r in coin_results]
        best_alphas = [r["best_alpha"] for r in coin_results]

        mean_imp = np.mean(improvements)
        max_imp = max(improvements)
        best_curve = max(coin_results, key=lambda r: r["improvement"])

        print(f"\n  {coin_name:14s}:")
        print(f"    Mean improvement: {mean_imp:.3f}x")
        print(f"    Max improvement:  {max_imp:.3f}x ({best_curve['curve']}, N={best_curve['N']})")
        print(f"    Best alphas:      {[f'{a:.3f}' for a in best_alphas]}")
        print(f"    Consistent alpha: ", end="")
        if len(set(best_alphas)) == 1:
            print(f"YES ({best_alphas[0]:.3f})")
        else:
            non_zero = [a for a in best_alphas if a > 0]
            if non_zero:
                print(f"no (range {min(best_alphas):.3f} - {max(best_alphas):.3f})")
            else:
                print(f"all zero (geometry never helps)")

    # ================================================================
    # ALPHA SWEEP DETAIL for best coin type
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  ALPHA SWEEP DETAIL")
    print(f"  {'='*74}")

    # Find which coin type has best overall improvement
    best_coin = max(all_results, key=lambda r: r["improvement"])
    print(f"\n  Best result: {best_coin['coin']} on {best_coin['curve']} "
          f"({best_coin['improvement']:.3f}x at alpha={best_coin['best_alpha']:.3f})")

    # Show alpha sweep for that coin across all curves
    target_coin = best_coin["coin"]
    print(f"\n  Alpha sweep for '{target_coin}' across all curves:")
    print(f"\n  {'Alpha':>8s}", end="")
    curve_names = sorted(set(r["curve"] for r in all_results))
    for cn in curve_names:
        print(f"  {cn:>10s}", end="")
    print()
    print(f"  {'-'*8}", end="")
    for _ in curve_names:
        print(f"  {'-'*10}", end="")
    print()

    for ai, alpha in enumerate(alphas):
        print(f"  {alpha:8.3f}", end="")
        for cn in curve_names:
            r = [x for x in all_results if x["coin"] == target_coin and x["curve"] == cn]
            if r and ai < len(r[0]["alpha_probs"]):
                prob = r[0]["alpha_probs"][ai]
                baseline = r[0]["baseline_prob"]
                ratio = prob / baseline if baseline > 0 else 0
                marker = " *" if ratio > 1.05 else ""
                print(f"  {ratio:8.3f}x{marker}", end="")
            else:
                print(f"  {'--':>10s}", end="")
        print()

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  VERDICT")
    print(f"  {'='*74}")

    any_signal = any(r["improvement"] > 1.1 for r in all_results)
    consistent = False

    if any_signal:
        signal_results = [r for r in all_results if r["improvement"] > 1.1]
        # Check if same coin type wins across multiple curves
        coin_counts = {}
        for r in signal_results:
            coin_counts[r["coin"]] = coin_counts.get(r["coin"], 0) + 1

        best_coin_name = max(coin_counts, key=coin_counts.get)
        best_count = coin_counts[best_coin_name]

        if best_count >= 3:
            consistent = True
            print(f"\n  CONSISTENT SIGNAL: '{best_coin_name}' improves walk on "
                  f"{best_count}/{len(curve_configs)} curves")
            for r in signal_results:
                if r["coin"] == best_coin_name:
                    print(f"    {r['curve']}: {r['improvement']:.3f}x at alpha={r['best_alpha']:.3f}")
        else:
            print(f"\n  SCATTERED SIGNAL: improvements found but not consistent")
            for r in signal_results:
                print(f"    {r['curve']}/{r['coin']}: {r['improvement']:.3f}x "
                      f"at alpha={r['best_alpha']:.3f}")
    else:
        print(f"\n  NO SIGNAL: No coin at any alpha beats Hadamard by >10%")
        print(f"  Geometry does not improve quantum walk on EC groups.")
        print(f"  The walk is insensitive to coordinate-space structure.")

    # ================================================================
    # WRITE CSV
    # ================================================================
    csv_path = "/Users/kjm/Desktop/quantum_walk_blend.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["curve", "N", "coin", "baseline_prob", "best_alpha",
                  "best_prob", "improvement"] + [f"alpha_{a}" for a in alphas]
        writer.writerow(header)
        for r in all_results:
            row = [r["curve"], r["N"], r["coin"],
                   f"{r['baseline_prob']:.8f}", r["best_alpha"],
                   f"{r['best_prob']:.8f}", f"{r['improvement']:.4f}"]
            row += [f"{p:.8f}" for p in r["alpha_probs"]]
            writer.writerow(row)

    print(f"\n  Results: {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
