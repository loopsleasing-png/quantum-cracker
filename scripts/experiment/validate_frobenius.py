"""Validate the Frobenius coin 419.9x result.

Critical question: does the frobenius coin CONSISTENTLY concentrate
probability at the target, or does it create peaks at random positions
that happened to align with our chosen target?

Tests:
1. Multi-target: sweep ALL positions as target, measure ratio at each
2. Shuffled control: randomize angle assignment, check if signal persists
3. Multi-coin comparison: compare all public-info coins fairly
4. Statistical summary: is the mean ratio across targets > 1.0?
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
    if p % 3 != 1:
        return None
    for g in range(2, p):
        beta = pow(g, (p - 1) // 3, p)
        if beta != 1 and pow(beta, 3, p) == 1:
            return beta
    return None


# ================================================================
# QUANTUM WALK ENGINE
# ================================================================

def run_walk(N, base_angle, geo_angles, alpha, start, target, max_steps):
    """Run walk and return max probability at target."""
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


def run_walk_full_distribution(N, base_angle, geo_angles, alpha, start, max_steps):
    """Run walk and return max probability at EVERY position."""
    blended = (1.0 - alpha) * base_angle + alpha * geo_angles
    cos_a = np.cos(blended)
    sin_a = np.sin(blended)

    left = np.zeros(N, dtype=np.complex128)
    right = np.zeros(N, dtype=np.complex128)
    left[start] = 1.0 / np.sqrt(2)
    right[start] = 1.0 / np.sqrt(2)

    max_probs = np.zeros(N)

    for step in range(1, max_steps + 1):
        new_left = cos_a * left + sin_a * right
        new_right = -sin_a * left + cos_a * right
        left = np.roll(new_left, -1)
        right = np.roll(new_right, 1)

        probs = np.abs(left)**2 + np.abs(right)**2
        max_probs = np.maximum(max_probs, probs)

    return max_probs


# ================================================================
# PUBLIC-INFO COINS
# ================================================================

def coin_frobenius(ec, cycle):
    N = len(cycle)
    p = ec.p
    trace = p + 1 - ec.order
    angles = np.full(N, np.pi / 4)
    for i, pt in enumerate(cycle):
        if pt is None: continue
        x, y = pt
        angles[i] = 2 * np.pi * ((x * abs(trace)) % p) / p
    return angles


def coin_endo_phase(ec, cycle, beta):
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)
    if beta is None: return angles
    for i, pt in enumerate(cycle):
        if pt is None: continue
        x, y = pt
        diff = (beta * x - x) % p
        angles[i] = np.pi * diff / p
    return angles


def coin_qr_legendre(ec, cycle):
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)
    for i, pt in enumerate(cycle):
        if pt is None: continue
        x, y = pt
        if x == 0: continue
        legendre = pow(x, (p - 1) // 2, p)
        angles[i] = np.pi / 3 if legendre == 1 else 2 * np.pi / 3
    return angles


def coin_y_parity(ec, cycle):
    N = len(cycle)
    angles = np.full(N, np.pi / 4)
    for i, pt in enumerate(cycle):
        if pt is None: continue
        x, y = pt
        angles[i] = np.pi / 3 if y % 2 == 0 else 2 * np.pi / 3
    return angles


def coin_x_cubic(ec, cycle):
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)
    for i, pt in enumerate(cycle):
        if pt is None: continue
        x, y = pt
        rhs = (x * x * x + ec.b) % p
        angles[i] = np.pi * rhs / p
    return angles


def coin_double_pt(ec, cycle):
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)
    for i, pt in enumerate(cycle):
        if pt is None: continue
        x, y = pt
        if y == 0: continue
        double = ec.add(pt, pt)
        if double is not None:
            angles[i] = 2 * np.pi * double[0] / p
    return angles


def coin_add_gen(ec, cycle):
    N = len(cycle)
    p = ec.p
    G = ec.generator
    angles = np.full(N, np.pi / 4)
    for i, pt in enumerate(cycle):
        pG = ec.add(pt, G) if pt is not None else G
        if pG is not None:
            angles[i] = 2 * np.pi * pG[0] / p
    return angles


def coin_div_poly(ec, cycle):
    N = len(cycle)
    p = ec.p
    angles = np.full(N, np.pi / 4)
    for i, pt in enumerate(cycle):
        if pt is None: continue
        x, y = pt
        psi3 = (3 * pow(x, 4, p) + 84 * x) % p
        angles[i] = 2 * np.pi * psi3 / p
    return angles


# ================================================================
# MAIN VALIDATION
# ================================================================

def main():
    print()
    print("=" * 78)
    print("  FROBENIUS COIN VALIDATION")
    print("  Testing whether 419.9x result is real or artifact")
    print("=" * 78)

    # Test on the curves that showed signal
    test_curves = [
        (7001, "p7001-endo"),   # 419.9x
        (2003, "p2003-ctrl"),   # 42.7x
        (2503, "p2503-endo"),   # endo_xratio 17.0x
        (1009, "p1009-endo"),   # curvature 1.7x
        (4003, "p4003-endo"),
    ]

    base_angle = np.pi / 4
    results = []

    for p_val, name in test_curves:
        print(f"\n{'='*78}")
        print(f"  CURVE: {name} (p={p_val})")
        print(f"{'='*78}")

        ec = SmallEC(p_val, 0, 7)
        gen = ec.find_generator()
        beta = find_cube_root_of_unity(p_val)
        N = ec.order

        # Build cycle
        cycle = []
        P = None
        for k in range(N):
            cycle.append(P)
            P = ec.add(P, gen)

        print(f"  |E| = {N}, beta = {beta}")

        # ============================================================
        # TEST 1: Full probability distribution (all targets)
        # ============================================================
        print(f"\n  TEST 1: Full probability distribution")
        print(f"  Running walk with frobenius coin (alpha=1.0) ...")

        frob_angles = coin_frobenius(ec, cycle)
        max_steps = min(N * 4, 5000)

        t0 = time.time()
        # Hadamard baseline: full distribution
        uniform = np.full(N, base_angle)
        had_dist = run_walk_full_distribution(N, base_angle, uniform, 0.0, 0, max_steps)

        # Frobenius: full distribution
        frob_dist = run_walk_full_distribution(N, base_angle, frob_angles, 1.0, 0, max_steps)
        dt = time.time() - t0

        # Compute ratio at every position
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(had_dist > 0, frob_dist / had_dist, 0)

        # Original target
        orig_target = (N // 3 + 7) % N
        orig_ratio = ratios[orig_target]

        print(f"  Time: {dt:.1f}s")
        print(f"  Original target (pos {orig_target}): ratio = {orig_ratio:.1f}x")
        print(f"  Max ratio across ALL positions: {ratios.max():.1f}x at pos {ratios.argmax()}")
        print(f"  Mean ratio: {ratios.mean():.2f}x")
        print(f"  Median ratio: {np.median(ratios):.2f}x")
        print(f"  Positions with ratio > 2x: {np.sum(ratios > 2)}/{N}")
        print(f"  Positions with ratio > 10x: {np.sum(ratios > 10)}/{N}")
        print(f"  Positions with ratio > 100x: {np.sum(ratios > 100)}/{N}")

        # Is the original target special, or just lucky?
        rank = np.sum(ratios >= orig_ratio)
        print(f"  Original target rank: {rank}/{N} (1 = best)")

        # ============================================================
        # TEST 2: Shuffled control
        # ============================================================
        print(f"\n  TEST 2: Shuffled control (break point-position mapping)")

        shuffled_angles = frob_angles.copy()
        np.random.seed(42)
        np.random.shuffle(shuffled_angles)

        shuf_dist = run_walk_full_distribution(N, base_angle, shuffled_angles, 1.0, 0, max_steps)
        with np.errstate(divide='ignore', invalid='ignore'):
            shuf_ratios = np.where(had_dist > 0, shuf_dist / had_dist, 0)

        print(f"  Shuffled max ratio: {shuf_ratios.max():.1f}x at pos {shuf_ratios.argmax()}")
        print(f"  Shuffled mean ratio: {shuf_ratios.mean():.2f}x")
        print(f"  Shuffled positions > 10x: {np.sum(shuf_ratios > 10)}/{N}")

        # ============================================================
        # TEST 3: Random angle control
        # ============================================================
        print(f"\n  TEST 3: Random angles control")

        np.random.seed(123)
        random_angles = np.random.uniform(0, 2 * np.pi, N)

        rand_dist = run_walk_full_distribution(N, base_angle, random_angles, 1.0, 0, max_steps)
        with np.errstate(divide='ignore', invalid='ignore'):
            rand_ratios = np.where(had_dist > 0, rand_dist / had_dist, 0)

        print(f"  Random max ratio: {rand_ratios.max():.1f}x at pos {rand_ratios.argmax()}")
        print(f"  Random mean ratio: {rand_ratios.mean():.2f}x")
        print(f"  Random positions > 10x: {np.sum(rand_ratios > 10)}/{N}")

        # ============================================================
        # TEST 4: All public-info coins comparison
        # ============================================================
        print(f"\n  TEST 4: All public-info coins (alpha=1.0)")

        coins = {
            "frobenius": frob_angles,
            "endo_phase": coin_endo_phase(ec, cycle, beta),
            "qr_legendre": coin_qr_legendre(ec, cycle),
            "y_parity": coin_y_parity(ec, cycle),
            "x_cubic": coin_x_cubic(ec, cycle),
            "double_pt": coin_double_pt(ec, cycle),
            "add_gen": coin_add_gen(ec, cycle),
            "div_poly": coin_div_poly(ec, cycle),
        }

        print(f"  {'Coin':<14} {'MaxRatio':>8} {'MeanRatio':>10} {'Pos>10x':>8} {'Pos>100x':>9}")
        print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*8} {'-'*9}")

        for cname, angles in coins.items():
            dist = run_walk_full_distribution(N, base_angle, angles, 1.0, 0, max_steps)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(had_dist > 0, dist / had_dist, 0)
            print(f"  {cname:<14} {r.max():>8.1f} {r.mean():>10.2f} {np.sum(r > 10):>8} {np.sum(r > 100):>9}")

            results.append({
                "curve": name,
                "N": N,
                "coin": cname,
                "max_ratio": f"{r.max():.2f}",
                "mean_ratio": f"{r.mean():.4f}",
                "pos_gt_10x": int(np.sum(r > 10)),
                "pos_gt_100x": int(np.sum(r > 100)),
            })

        # Shuffled and random for comparison
        for tag, angles in [("SHUFFLED", shuffled_angles), ("RANDOM", random_angles)]:
            dist = run_walk_full_distribution(N, base_angle, angles, 1.0, 0, max_steps)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(had_dist > 0, dist / had_dist, 0)
            print(f"  {tag:<14} {r.max():>8.1f} {r.mean():>10.2f} {np.sum(r > 10):>8} {np.sum(r > 100):>9}")

        # ============================================================
        # TEST 5: Multiple start positions
        # ============================================================
        print(f"\n  TEST 5: Multiple start positions (frobenius coin)")

        start_positions = [0, N // 4, N // 3, N // 2, 2 * N // 3]
        for start in start_positions:
            dist = run_walk_full_distribution(N, base_angle, frob_angles, 1.0, start, max_steps)
            with np.errstate(divide='ignore', invalid='ignore'):
                had_start_dist = run_walk_full_distribution(N, base_angle, uniform, 0.0, start, max_steps)
                r = np.where(had_start_dist > 0, dist / had_start_dist, 0)
            print(f"  Start={start:>5}: max_ratio={r.max():.1f}x, mean={r.mean():.2f}x, pos>10x={np.sum(r > 10)}")

    # Save results
    csv_path = f"{sys.path[0]}/../scripts/experiment/validate_frobenius_results.csv"
    csv_desktop = f"/Users/kjm/Desktop/validate_frobenius.csv"
    for path in [csv_desktop]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["curve", "N", "coin", "max_ratio", "mean_ratio", "pos_gt_10x", "pos_gt_100x"])
            w.writeheader()
            w.writerows(results)
        print(f"\n  Results saved: {path}")

    # ============================================================
    # VERDICT
    # ============================================================
    print(f"\n{'='*78}")
    print(f"  VERDICT")
    print(f"{'='*78}")
    print(f"""
  If frobenius max_ratio >> shuffled max_ratio AND >> random max_ratio:
    -> The ORDERING of angles along the cycle matters
    -> The EC group structure creates real interference
    -> But this still may not help: peaks at UNKNOWN positions

  If frobenius max_ratio ~ shuffled max_ratio ~ random max_ratio:
    -> ANY non-uniform coin creates peaks (generic QW property)
    -> Frobenius structure adds nothing over random angles
    -> The 419.9x was not special

  For a coin to be USEFUL for DLP:
    -> It must concentrate probability at the TARGET specifically
    -> This means the coin must encode target-relative information
    -> Public-info coins can't do this (they don't know the target's position)
    """)


if __name__ == "__main__":
    main()
