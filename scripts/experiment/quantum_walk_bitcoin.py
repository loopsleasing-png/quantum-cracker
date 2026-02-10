"""Quantum Walk with Bitcoin-Specific Coin Designs.

secp256k1 (Bitcoin's curve) has special structure that generic curves don't:

  1. ENDOMORPHISM: y^2 = x^3 + 7 has j-invariant = 0 (because a=0).
     This gives an efficient endomorphism: (x, y) -> (beta*x, y)
     where beta is a cube root of unity mod p (beta^3 = 1).
     Bitcoin uses this (GLV method) for 30% faster signature verification.
     This is REAL algebraic structure specific to this curve.

  2. CUBE ROOT SYMMETRY: The map x -> beta*x permutes curve points
     in groups of 3. This creates a 3-fold symmetry on the curve
     that doesn't exist on generic curves (where a != 0).

  3. QUADRATIC TWIST: For each x, either (x, y) is on the curve or
     (x, y') is on the "twist" curve. The distribution of curve vs
     twist points encodes information about the field.

  4. FROBENIUS TRACE: The group order |E| = p + 1 - t. The trace t
     encodes deep arithmetic of the curve. For secp256k1 t is specific.

  5. DIVISION POLYNOMIALS: The n-division polynomial psi_n has roots at
     points P where nP = O (n-torsion). These create special subgroups.

We build quantum walk coins from each of these structures and test
whether any creates constructive interference at the DLP target.

Key insight: the cycle_pos coin worked (10.7x) because it encodes
group structure. These Bitcoin-specific coins encode PARTIAL group
structure available from PUBLIC information only.
"""

import csv
import math
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, "src")


# ================================================================
# SMALL EC (reused)
# ================================================================

class SmallEC:
    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b
        self.points = self._enumerate_points()
        self.order = len(self.points)
        self.generator = None

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
        if x1 == x2 and y1 == (p - y2) % p: return None
        if P == Q:
            if y1 == 0: return None
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, p - 2, p) % p
        else:
            if x1 == x2: return None
            lam = (y2 - y1) * pow((x2 - x1) % p, p - 2, p) % p
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
                            is_gen = False; break
                if is_gen:
                    self.generator = pt; return pt
        self.generator = self.points[1]
        return self.generator


def find_cube_root_of_unity(p):
    """Find beta such that beta^3 = 1 mod p, beta != 1.

    Exists iff p = 1 mod 3.
    """
    if p % 3 != 1:
        return None
    # Find a generator of the multiplicative group
    # beta = g^((p-1)/3) for any generator g
    for g in range(2, p):
        beta = pow(g, (p - 1) // 3, p)
        if beta != 1 and pow(beta, 3, p) == 1:
            return beta
    return None


# ================================================================
# VECTORIZED QUANTUM WALK (same engine)
# ================================================================

def run_blended_walk(N, base_angle, geo_angles, alpha, start, target, max_steps):
    blended = (1.0 - alpha) * base_angle + alpha * geo_angles
    cos_a = np.cos(blended)
    sin_a = np.sin(blended)

    left = np.zeros(N, dtype=np.complex128)
    right = np.zeros(N, dtype=np.complex128)
    left[start] = 1.0 / np.sqrt(2)
    right[start] = 1.0 / np.sqrt(2)

    max_prob = abs(left[target])**2 + abs(right[target])**2
    max_step = 0

    for step in range(1, max_steps + 1):
        new_left = cos_a * left + sin_a * right
        new_right = -sin_a * left + cos_a * right
        left = np.roll(new_left, -1)
        right = np.roll(new_right, 1)

        p_target = abs(left[target])**2 + abs(right[target])**2
        if p_target > max_prob:
            max_prob = p_target
            max_step = step

    return max_prob, max_step


# ================================================================
# BITCOIN-SPECIFIC COIN DESIGNS
# ================================================================

def coin_endomorphism(ec, cycle, beta):
    """Coin from secp256k1 endomorphism: (x,y) -> (beta*x, y).

    The angle encodes the DISTANCE between a point and its
    endomorphic image in the group cycle. This distance is
    public information (computable from coordinates).
    """
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)

    # Build point -> cycle index map
    pt_to_idx = {}
    for i, pt in enumerate(cycle):
        if pt is not None:
            pt_to_idx[pt] = i

    for i, pt in enumerate(cycle):
        if pt is None:
            continue
        x, y = pt
        # Endomorphic image
        endo_pt = ((beta * x) % p, y)
        if endo_pt in pt_to_idx:
            # Distance in cycle between point and its endo-image
            j = pt_to_idx[endo_pt]
            dist = (j - i) % N
            angles[i] = 2 * np.pi * dist / N
        # If endo_pt not on curve (shouldn't happen), keep default

    return angles


def coin_endo_phase(ec, cycle, beta):
    """Phase difference between point and its endomorphic image.

    Instead of cycle distance, use the COORDINATE difference.
    This is purely public information.
    """
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)

    for i, pt in enumerate(cycle):
        if pt is None:
            continue
        x, y = pt
        bx = (beta * x) % p
        # Phase from coordinate difference
        diff = (bx - x) % p
        angles[i] = np.pi * diff / p

    return angles


def coin_endo_xratio(ec, cycle, beta):
    """Ratio x / (beta*x) mod p as phase.

    This encodes how the endomorphism stretches the x-coordinate.
    """
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)

    for i, pt in enumerate(cycle):
        if pt is None:
            continue
        x, y = pt
        if x == 0:
            continue
        bx = (beta * x) % p
        # x / bx mod p
        ratio = (x * pow(bx, p - 2, p)) % p
        angles[i] = 2 * np.pi * ratio / p

    return angles


def coin_threefold(ec, cycle, beta):
    """3-fold orbit structure.

    Each point P has orbit {P, endo(P), endo(endo(P))}.
    Coin angle = which position in the orbit (0, 1, or 2).
    This creates a 3-periodic structure in the walk.
    """
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)

    pt_to_idx = {}
    for i, pt in enumerate(cycle):
        if pt is not None:
            pt_to_idx[pt] = i

    visited = set()
    for i, pt in enumerate(cycle):
        if pt is None or i in visited:
            continue
        x, y = pt
        # Build orbit
        orbit = [i]
        visited.add(i)
        cur = pt
        for _ in range(2):
            cur = ((beta * cur[0]) % p, cur[1])
            if cur in pt_to_idx:
                j = pt_to_idx[cur]
                orbit.append(j)
                visited.add(j)

        # Assign phases: 0, 2pi/3, 4pi/3
        for pos, idx in enumerate(orbit):
            angles[idx] = 2 * np.pi * pos / 3

    return angles


def coin_quadratic_residue(ec, cycle):
    """Coin from quadratic residuosity of x-coordinate.

    Legendre symbol (x/p) = +1 or -1 for most x.
    This splits curve points into two classes: QR and QNR x-values.
    """
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)

    for i, pt in enumerate(cycle):
        if pt is None:
            continue
        x, y = pt
        if x == 0:
            continue
        # Legendre symbol via Euler's criterion
        legendre = pow(x, (p - 1) // 2, p)
        if legendre == 1:
            angles[i] = np.pi / 3  # QR
        else:
            angles[i] = 2 * np.pi / 3  # QNR

    return angles


def coin_y_parity(ec, cycle):
    """Coin from y-coordinate parity (even/odd).

    This is actually used in Bitcoin's compressed public key format.
    The "02" vs "03" prefix encodes y parity.
    """
    N = len(cycle)
    angles = np.full(N, np.pi / 4)

    for i, pt in enumerate(cycle):
        if pt is None:
            continue
        x, y = pt
        if y % 2 == 0:
            angles[i] = np.pi / 3
        else:
            angles[i] = 2 * np.pi / 3

    return angles


def coin_x_cubic(ec, cycle):
    """Coin from x^3 + 7 mod p (the RHS of the curve equation).

    This is the "shape" of the curve at each point.
    For y^2 = x^3 + 7, the RHS = y^2, so this is equivalent to |y|.
    """
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)

    for i, pt in enumerate(cycle):
        if pt is None:
            continue
        x, y = pt
        rhs = (x * x * x + ec.b) % p
        angles[i] = np.pi * rhs / p

    return angles


def coin_frobenius_trace(ec, cycle):
    """Coin encoding the Frobenius trace structure.

    The trace t = p + 1 - |E|. Points where the Frobenius action
    has specific eigenvalue structure get different phases.
    We approximate by using (x^2 mod |E|) as a Frobenius proxy.
    """
    N = len(cycle)
    p = ec.p
    order = ec.order
    trace = p + 1 - order
    angles = np.full(N, np.pi / 4)

    for i, pt in enumerate(cycle):
        if pt is None:
            continue
        x, y = pt
        # Frobenius-related: x^p mod p = x (trivially), but
        # the Frobenius action on the curve group is richer.
        # Use trace to create phase: (x * trace) mod p
        fro_phase = (x * abs(trace)) % p
        angles[i] = 2 * np.pi * fro_phase / p

    return angles


def coin_division_poly(ec, cycle):
    """Coin from 2-division polynomial structure.

    The 2-division polynomial for y^2 = x^3 + 7 is:
    psi_2(x) = 2y, psi_3(x) = 3x^4 + 12*7*x = 3x^4 + 84x (for a=0, b=7)

    Points where psi_3 is close to zero are 3-torsion points.
    Proximity to torsion encodes group structure.
    """
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)

    for i, pt in enumerate(cycle):
        if pt is None:
            continue
        x, y = pt
        # 3-division polynomial value (simplified for a=0)
        psi3 = (3 * pow(x, 4, p) + 84 * x) % p
        # Normalize to angle
        angles[i] = 2 * np.pi * psi3 / p

    return angles


def coin_double_point(ec, cycle):
    """Coin from the doubling map: P -> 2P.

    The x-coordinate of 2P is computable from P's coordinates.
    This encodes the curve's group law into the walk.
    """
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)

    pt_to_idx = {}
    for i, pt in enumerate(cycle):
        if pt is not None:
            pt_to_idx[pt] = i

    for i, pt in enumerate(cycle):
        if pt is None:
            continue
        x, y = pt
        if y == 0:
            continue
        # 2P
        double = ec.add(pt, pt)
        if double is None:
            angles[i] = 0
        elif double in pt_to_idx:
            j = pt_to_idx[double]
            # Use the x-coordinate of 2P as phase (PUBLIC info)
            dx = double[0]
            angles[i] = 2 * np.pi * dx / p

    return angles


def coin_add_generator(ec, cycle):
    """Coin from adding the generator: P -> P + G.

    The x-coordinate of P + G is computable from P and G (public).
    This encodes how the group operation moves through coordinate space.
    """
    N = len(cycle)
    p = ec.p
    G = ec.generator
    angles = np.full(N, np.pi / 4)

    for i, pt in enumerate(cycle):
        if pt is None:
            pG = G
        else:
            pG = ec.add(pt, G)

        if pG is None:
            angles[i] = 0
        else:
            angles[i] = 2 * np.pi * pG[0] / p

    return angles


# ================================================================
# MAIN
# ================================================================

def main():
    print()
    print("=" * 78)
    print("  QUANTUM WALK -- BITCOIN-SPECIFIC COIN DESIGNS")
    print("  Exploiting secp256k1 structure: endomorphism, j=0, torsion")
    print("=" * 78)

    # Use primes where p ≡ 1 mod 3 (endomorphism exists) AND p ≡ 2 mod 3 (control)
    curve_configs = [
        # p ≡ 1 mod 3 (endomorphism available)
        (211, "p211-endo"),
        (307, "p307-endo"),
        (601, "p601-endo"),
        (1009, "p1009-endo"),
        (2503, "p2503-endo"),
        (4003, "p4003-endo"),
        (7001, "p7001-endo"),
        # p ≡ 2 mod 3 (no endomorphism -- control)
        (251, "p251-ctrl"),
        (503, "p503-ctrl"),
        (2003, "p2003-ctrl"),
        (8191, "p8191-ctrl"),
    ]

    # Alpha values to sweep (focused on the 0-0.3 range where signal lives)
    alphas = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

    base_angle = np.pi / 4

    print(f"\n  Building curves (all y^2 = x^3 + 7)...")
    curves = {}
    for p_val, name in curve_configs:
        ec = SmallEC(p_val, 0, 7)
        gen = ec.find_generator()
        if gen is None:
            continue

        beta = find_cube_root_of_unity(p_val)
        has_endo = beta is not None

        cycle = []
        P = None
        for k in range(ec.order):
            cycle.append(P)
            P = ec.add(P, gen)

        curves[name] = {"ec": ec, "cycle": cycle, "beta": beta, "N": len(cycle)}
        endo_str = f"beta={beta}" if has_endo else "no endo"
        print(f"    {name}: |E|={ec.order}, {endo_str}")

    # ================================================================
    # RUN EXPERIMENTS
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  EXPERIMENTS: {len(alphas)} alphas x 11 coin types x {len(curves)} curves")
    print(f"  {'='*74}")

    all_results = []

    for name, cdata in curves.items():
        ec = cdata["ec"]
        cycle = cdata["cycle"]
        beta = cdata["beta"]
        N = cdata["N"]

        if N < 10:
            continue

        target_k = N // 3 + 7
        target_k = target_k % N
        max_steps = min(N * 4, 5000)

        # Baseline
        uniform = np.full(N, base_angle)
        baseline, baseline_step = run_blended_walk(
            N, base_angle, uniform, 0.0, 0, target_k, max_steps)

        print(f"\n  {name}: N={N}, baseline={baseline:.6f}")

        # Build all coin angle arrays
        coin_defs = [
            ("qr_legendre", coin_quadratic_residue(ec, cycle)),
            ("y_parity", coin_y_parity(ec, cycle)),
            ("x_cubic", coin_x_cubic(ec, cycle)),
            ("frobenius", coin_frobenius_trace(ec, cycle)),
            ("div_poly_3", coin_division_poly(ec, cycle)),
            ("double_pt", coin_double_point(ec, cycle)),
            ("add_gen", coin_add_generator(ec, cycle)),
        ]

        # Add endomorphism coins only if beta exists
        if beta is not None:
            coin_defs.extend([
                ("endo_dist", coin_endomorphism(ec, cycle, beta)),
                ("endo_phase", coin_endo_phase(ec, cycle, beta)),
                ("endo_xratio", coin_endo_xratio(ec, cycle, beta)),
                ("threefold", coin_threefold(ec, cycle, beta)),
            ])

        for coin_name, geo_angles in coin_defs:
            best_alpha = 0.0
            best_prob = baseline
            best_step = baseline_step
            sweep = []

            t0 = time.time()
            for alpha in alphas:
                mp, ms = run_blended_walk(
                    N, base_angle, geo_angles, alpha, 0, target_k, max_steps)
                sweep.append(mp)
                if mp > best_prob:
                    best_prob = mp
                    best_alpha = alpha
                    best_step = ms
            elapsed = time.time() - t0

            improvement = best_prob / baseline if baseline > 0 else 0
            marker = " ***" if improvement > 1.1 else ""

            print(f"    {coin_name:14s}: alpha={best_alpha:.3f} "
                  f"P={best_prob:.6f} ({improvement:.2f}x){marker}  "
                  f"[{elapsed:.1f}s]")

            all_results.append({
                "curve": name,
                "N": N,
                "coin": coin_name,
                "has_endo": beta is not None,
                "baseline": baseline,
                "best_alpha": best_alpha,
                "best_prob": best_prob,
                "improvement": improvement,
                "sweep": sweep,
            })

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  ANALYSIS: Which Bitcoin-specific coins show signal?")
    print(f"  {'='*74}")

    # Group by coin type, show mean improvement
    coin_names = sorted(set(r["coin"] for r in all_results))

    print(f"\n  {'Coin':>14s}  {'Mean':>6s}  {'Max':>6s}  {'Curves>1.1x':>12s}  "
          f"{'Best Curve':>12s}  {'Best Alpha':>11s}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*11}")

    signal_coins = []
    for cn in coin_names:
        cr = [r for r in all_results if r["coin"] == cn]
        imps = [r["improvement"] for r in cr]
        above = sum(1 for i in imps if i > 1.1)
        best = max(cr, key=lambda r: r["improvement"])

        if max(imps) > 1.1:
            signal_coins.append(cn)

        print(f"  {cn:>14s}  {np.mean(imps):5.2f}x  {max(imps):5.2f}x  "
              f"{above:8d}/{len(cr)}  {best['curve']:>12s}  {best['best_alpha']:11.3f}")

    # ================================================================
    # ENDOMORPHISM vs NON-ENDOMORPHISM CURVES
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  ENDOMORPHISM ANALYSIS: Do endo curves behave differently?")
    print(f"  {'='*74}")

    endo_results = [r for r in all_results if r["has_endo"]]
    ctrl_results = [r for r in all_results if not r["has_endo"]]

    if endo_results and ctrl_results:
        endo_imps = [r["improvement"] for r in endo_results]
        ctrl_imps = [r["improvement"] for r in ctrl_results]

        print(f"\n  Endomorphism curves (p ≡ 1 mod 3):")
        print(f"    Mean improvement: {np.mean(endo_imps):.4f}x")
        print(f"    Max improvement:  {max(endo_imps):.4f}x")
        print(f"    Coins above 1.1x: {sum(1 for i in endo_imps if i > 1.1)}/{len(endo_imps)}")

        print(f"\n  Control curves (p ≡ 2 mod 3):")
        print(f"    Mean improvement: {np.mean(ctrl_imps):.4f}x")
        print(f"    Max improvement:  {max(ctrl_imps):.4f}x")
        print(f"    Coins above 1.1x: {sum(1 for i in ctrl_imps if i > 1.1)}/{len(ctrl_imps)}")

        # t-test
        t_stat, p_val = stats.ttest_ind(endo_imps, ctrl_imps)
        print(f"\n  t-test (endo vs ctrl): t={t_stat:.3f}, p={p_val:.6f}")
        if p_val < 0.05:
            print(f"  *** SIGNIFICANT: Endomorphism curves respond differently ***")
        else:
            print(f"  No significant difference between endo and control curves")

    # ================================================================
    # BEST COIN DETAIL
    # ================================================================
    if signal_coins:
        print(f"\n  {'='*74}")
        print(f"  DETAIL: Alpha sweep for coins with signal")
        print(f"  {'='*74}")

        for cn in signal_coins:
            cr = [r for r in all_results if r["coin"] == cn]
            print(f"\n  {cn}:")
            print(f"  {'Alpha':>8s}", end="")
            for r in cr:
                print(f"  {r['curve'][:8]:>10s}", end="")
            print()

            for ai, alpha in enumerate(alphas):
                print(f"  {alpha:8.3f}", end="")
                for r in cr:
                    if ai < len(r["sweep"]):
                        ratio = r["sweep"][ai] / r["baseline"] if r["baseline"] > 0 else 0
                        m = " *" if ratio > 1.05 else ""
                        print(f"  {ratio:8.3f}x{m}", end="")
                print()

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  VERDICT")
    print(f"  {'='*74}")

    consistent_coins = []
    for cn in coin_names:
        cr = [r for r in all_results if r["coin"] == cn]
        above = sum(1 for r in cr if r["improvement"] > 1.1)
        if above >= len(cr) // 2:  # majority of curves
            consistent_coins.append((cn, above, len(cr)))

    if consistent_coins:
        print(f"\n  SIGNAL DETECTED in {len(consistent_coins)} coin type(s):")
        for cn, above, total in consistent_coins:
            best_r = max([r for r in all_results if r["coin"] == cn],
                        key=lambda r: r["improvement"])
            print(f"    {cn}: {above}/{total} curves above 1.1x, "
                  f"peak={best_r['improvement']:.2f}x at alpha={best_r['best_alpha']:.3f}")
    else:
        any_signal = [r for r in all_results if r["improvement"] > 1.1]
        if any_signal:
            print(f"\n  SCATTERED SIGNAL: {len(any_signal)} individual results above 1.1x")
            print(f"  but no coin type consistently beats Hadamard across curves.")
            for r in sorted(any_signal, key=lambda r: -r["improvement"])[:10]:
                print(f"    {r['coin']:14s} on {r['curve']:12s}: "
                      f"{r['improvement']:.2f}x at alpha={r['best_alpha']:.3f}")
        else:
            print(f"\n  NO SIGNAL: No Bitcoin-specific coin improves the quantum walk.")
            print(f"  The algebraic structure of secp256k1 does not create")
            print(f"  constructive interference in discrete-time quantum walks.")

    # ================================================================
    # WRITE CSV
    # ================================================================
    csv_path = "/Users/kjm/Desktop/quantum_walk_bitcoin.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["curve", "N", "coin", "has_endo", "baseline",
                  "best_alpha", "best_prob", "improvement"]
        header += [f"alpha_{a}" for a in alphas]
        writer.writerow(header)
        for r in all_results:
            row = [r["curve"], r["N"], r["coin"], r["has_endo"],
                   f"{r['baseline']:.8f}", r["best_alpha"],
                   f"{r['best_prob']:.8f}", f"{r['improvement']:.4f}"]
            row += [f"{s:.8f}" for s in r["sweep"]]
            writer.writerow(row)

    print(f"\n  Results: {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
