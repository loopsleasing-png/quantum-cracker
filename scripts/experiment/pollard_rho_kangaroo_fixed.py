"""Pollard's Rho & Kangaroo -- Fixed, Benchmarked Implementations.

The previous dlp_algorithm_battery.py had a buggy Pollard rho that
failed on all curves (wrong group order: used N-1 instead of the
actual subgroup order, and no proper restart logic). This script
provides correct, well-tested implementations.

Algorithms:
1. Baby-step Giant-step (BSGS) -- O(sqrt(n)) baseline
2. Pollard's rho -- Floyd's cycle, 3-partition, restart on db=0
3. Pollard's kangaroo (lambda) -- bounded interval with distinguished points

All tested on y^2 = x^3 + 7 (secp256k1 family) over small primes.
Scaling analysis extrapolates to secp256k1.
"""

import csv
import math
import os
import secrets
import sys
import time

import numpy as np

sys.path.insert(0, "src")


# ================================================================
# ELLIPTIC CURVE ARITHMETIC
# ================================================================

class SmallEC:
    """Elliptic curve y^2 = x^3 + ax + b over F_p.

    Standalone implementation with point enumeration for small curves.
    """

    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b
        self._points = None
        self._order = None
        self._gen = None

    @property
    def order(self):
        """Group order |E(F_p)| including the point at infinity."""
        if self._order is None:
            self._enumerate()
        return self._order

    @property
    def generator(self):
        """Find a generator of the group (or largest prime-order subgroup)."""
        if self._gen is None:
            self._find_generator()
        return self._gen

    def _enumerate(self):
        """Enumerate all points on the curve by brute force."""
        points = [None]  # point at infinity
        p, a, b = self.p, self.a, self.b
        # Precompute quadratic residues
        qr = {}
        for y in range(p):
            qr.setdefault((y * y) % p, []).append(y)
        for x in range(p):
            rhs = (x * x * x + a * x + b) % p
            if rhs in qr:
                for y in qr[rhs]:
                    points.append((x, y))
        self._points = points
        self._order = len(points)

    def _find_generator(self):
        """Find a generator by checking point orders against group order."""
        if self._points is None:
            self._enumerate()
        n = self.order
        # Factor n to check generator condition:
        # P is a generator iff P has order n, i.e., for each prime factor q|n,
        # (n/q)*P != infinity
        factors = _prime_factors(n)
        for pt in self._points[1:]:
            if self.multiply(pt, n) is not None:
                continue  # order doesn't divide n (shouldn't happen)
            is_gen = True
            for q in factors:
                if self.multiply(pt, n // q) is None:
                    is_gen = False
                    break
            if is_gen:
                self._gen = pt
                return pt
        # Fallback: return first non-infinity point
        self._gen = self._points[1] if len(self._points) > 1 else None
        return self._gen

    def add(self, P, Q):
        """Add two points on the curve."""
        if P is None:
            return Q
        if Q is None:
            return P
        p = self.p
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and y1 == (p - y2) % p:
            return None  # P + (-P) = O
        if P == Q:
            if y1 == 0:
                return None
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, p - 2, p) % p
        else:
            if x1 == x2:
                return None
            lam = (y2 - y1) * pow((x2 - x1) % p, p - 2, p) % p
        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def neg(self, P):
        """Negate a point."""
        if P is None:
            return None
        return (P[0], (self.p - P[1]) % self.p)

    def multiply(self, P, k):
        """Scalar multiplication using double-and-add."""
        if k < 0:
            P = self.neg(P)
            k = -k
        if k == 0 or P is None:
            return None
        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result


def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _factor(n):
    """Trial division factorization. Returns list of (prime, exponent)."""
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                e += 1
                n //= d
            factors.append((d, e))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors


def _subgroup_order(ec, G):
    """Compute the order of point G in the group.

    For a generator of the full group, this equals ec.order.
    For small curves we can just check.
    """
    n = ec.order
    # The order of G divides n. Find the smallest k > 0 with kG = O.
    # Check divisors of n in increasing order.
    divs = sorted(_divisors(n))
    for d in divs:
        if d > 0 and ec.multiply(G, d) is None:
            return d
    return n


def _divisors(n):
    """Return all positive divisors of n."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return divs


# ================================================================
# ALGORITHM 1: BABY-STEP GIANT-STEP (BSGS)
# ================================================================

def baby_step_giant_step(ec, G, Q, group_order=None):
    """Shanks' BSGS. O(sqrt(n)) time and space.

    Returns (key_found, total_operations).
    group_order: the order of the cyclic group generated by G.
    """
    if group_order is None:
        group_order = _subgroup_order(ec, G)
    n = group_order
    if n <= 1:
        return 0, 0

    m = int(math.isqrt(n)) + 1
    ops = 0

    # Baby steps: j -> j*G for j in [0, m)
    baby = {}
    P = None  # 0*G = O
    for j in range(m):
        key = P if P is not None else "inf"
        baby[key] = j
        P = ec.add(P, G)
        ops += 1

    # Precompute -m*G
    mG = ec.multiply(G, m)
    neg_mG = ec.neg(mG)
    ops += int(math.log2(max(m, 1))) + 1

    # Giant steps: Q - i*m*G for i in [0, m)
    gamma = Q
    for i in range(m):
        key = gamma if gamma is not None else "inf"
        if key in baby:
            k = (baby[key] + i * m) % n
            if ec.multiply(G, k) == Q:
                return k, ops
            # Edge case: try without modular reduction
            k_raw = baby[key] + i * m
            if k_raw != k and ec.multiply(G, k_raw) == Q:
                return k_raw, ops
        gamma = ec.add(gamma, neg_mG)
        ops += 1

    return None, ops


# ================================================================
# ALGORITHM 2: POLLARD'S RHO (FIXED)
# ================================================================

def pollard_rho(ec, G, Q, group_order=None, max_restarts=5):
    """Pollard's rho for ECDLP with Floyd's cycle detection.

    Uses the standard 3-partition iteration from Pollard's original paper.
    Each walk state is (R, a, b) where R = a*G + b*Q.
    On collision R_tortoise == R_hare, solve:
        (a_t - a_h) * G = (b_h - b_t) * Q
        => k = (a_t - a_h) * (b_h - b_t)^{-1} mod n

    Returns (key_found, total_operations).
    """
    if group_order is None:
        group_order = _subgroup_order(ec, G)
    n = group_order
    if n <= 1:
        return 0, 0

    # Check for trivial case Q = O
    if Q is None:
        return 0, 0

    # Check Q = G
    if Q == G:
        return 1, 0

    total_ops = 0
    max_iter_per_restart = 4 * int(math.isqrt(n)) + 100

    def partition(P):
        """3-partition based on x-coordinate."""
        if P is None:
            return 0
        return P[0] % 3

    def step(R, a, b):
        """One iteration of the random walk.

        Partition 0: R -> R + Q,    a unchanged, b -> b + 1
        Partition 1: R -> 2R,       a -> 2a,     b -> 2b
        Partition 2: R -> R + G,    a -> a + 1,  b unchanged
        """
        s = partition(R)
        if s == 0:
            R = ec.add(R, Q)
            b = (b + 1) % n
        elif s == 1:
            R = ec.add(R, R)
            a = (a * 2) % n
            b = (b * 2) % n
        else:
            R = ec.add(R, G)
            a = (a + 1) % n
        return R, a, b

    for restart in range(max_restarts):
        # Random starting point: R = a*G + b*Q
        a0 = secrets.randbelow(n) if n > 1 else 0
        b0 = secrets.randbelow(n) if n > 1 else 0
        # Ensure we don't start at infinity with both zero
        if a0 == 0 and b0 == 0:
            a0 = 1
        R0 = ec.add(ec.multiply(G, a0), ec.multiply(Q, b0))
        total_ops += 2  # two multiplications for starting point

        # Tortoise and hare
        a_t, b_t, R_t = a0, b0, R0
        a_h, b_h, R_h = a0, b0, R0

        for iteration in range(max_iter_per_restart):
            # Tortoise: one step
            R_t, a_t, b_t = step(R_t, a_t, b_t)
            total_ops += 1

            # Hare: two steps
            R_h, a_h, b_h = step(R_h, a_h, b_h)
            R_h, a_h, b_h = step(R_h, a_h, b_h)
            total_ops += 2

            if R_t == R_h:
                # Collision found: a_t*G + b_t*Q = a_h*G + b_h*Q
                # => (a_t - a_h)*G = (b_h - b_t)*Q
                # => k = (a_t - a_h) / (b_h - b_t) mod n
                db = (b_h - b_t) % n
                da = (a_t - a_h) % n

                if db == 0:
                    # Useless collision (same b coefficient).
                    # Must restart with new random start.
                    break

                # Try to compute k = da * db^{-1} mod n
                g = math.gcd(db, n)
                if g == 1:
                    # n is effectively prime relative to db; direct inverse
                    try:
                        db_inv = pow(db, -1, n)
                        k = (da * db_inv) % n
                        if ec.multiply(G, k) == Q:
                            return k, total_ops
                    except (ValueError, ZeroDivisionError):
                        break
                else:
                    # gcd(db, n) > 1: n is not prime or db shares a factor.
                    # There are g possible solutions: k = da/g * (db/g)^{-1} + j*(n/g)
                    if da % g != 0:
                        # No solution in this collision
                        break
                    da_red = da // g
                    db_red = db // g
                    n_red = n // g
                    try:
                        db_red_inv = pow(db_red, -1, n_red)
                    except (ValueError, ZeroDivisionError):
                        break
                    base_k = (da_red * db_red_inv) % n_red
                    for j in range(g):
                        k_candidate = (base_k + j * n_red) % n
                        if ec.multiply(G, k_candidate) == Q:
                            return k_candidate, total_ops
                    # None of the candidates worked
                    break

    return None, total_ops


# ================================================================
# ALGORITHM 3: POLLARD'S KANGAROO (LAMBDA METHOD)
# ================================================================

def pollard_kangaroo(ec, G, Q, lo, hi, group_order=None):
    """Pollard's kangaroo (lambda) for ECDLP when k is in [lo, hi].

    Tame kangaroo starts at hi*G and jumps forward.
    Wild kangaroo starts at Q and jumps forward.
    When wild lands on a point the tame visited, we can deduce k.

    Uses distinguished points for memory-efficient collision detection.

    Returns (key_found, total_operations).
    """
    if group_order is None:
        group_order = _subgroup_order(ec, G)
    n = group_order

    interval = hi - lo
    if interval <= 0:
        # Check if Q == lo*G
        if ec.multiply(G, lo % n) == Q:
            return lo % n, 1
        return None, 1

    total_ops = 0

    # Jump set: powers of 2 up to roughly sqrt(interval)
    # The mean step size should be about sqrt(interval) / 4 for optimal performance
    num_jumps = max(4, int(math.log2(max(interval, 2))) // 2 + 1)
    jump_sizes = [1 << i for i in range(num_jumps)]
    mean_step = sum(jump_sizes) / len(jump_sizes)

    # Precompute jump point multiples: s_i * G for each jump size
    jump_points = {}
    for s in jump_sizes:
        jump_points[s] = ec.multiply(G, s)
        total_ops += 1

    def jump_index(P):
        """Deterministic jump selection based on point."""
        if P is None:
            return 0
        return P[0] % num_jumps

    # Distinguished point criterion: x-coordinate has dp_bits trailing zeros
    # Expected steps between DPs = 2^dp_bits
    # We want this to be roughly sqrt(interval) / 4
    dp_bits = max(1, int(math.log2(max(interval, 4))) // 4)
    dp_mask = (1 << dp_bits) - 1

    def is_distinguished(P):
        if P is None:
            return True
        return (P[0] & dp_mask) == 0

    # Number of tame steps: about 4 * sqrt(interval) / mean_step
    # but bounded to not be astronomical
    max_tame_steps = int(4 * math.sqrt(interval)) + 100
    max_wild_steps = int(4 * math.sqrt(interval)) + 100

    # --- Tame kangaroo ---
    # Starts at b*G, accumulates distance d_T
    T = ec.multiply(G, hi % n)
    total_ops += 1
    d_T = 0
    tame_traps = {}  # point -> accumulated_distance

    for _ in range(max_tame_steps):
        ji = jump_index(T)
        s = jump_sizes[ji]
        T = ec.add(T, jump_points[s])
        d_T += s
        total_ops += 1

        if is_distinguished(T):
            key = T if T is not None else "inf"
            tame_traps[key] = d_T

    # --- Wild kangaroo ---
    # Starts at Q = k*G, accumulates distance d_W
    W = Q
    d_W = 0

    for _ in range(max_wild_steps):
        ji = jump_index(W)
        s = jump_sizes[ji]
        W = ec.add(W, jump_points[s])
        d_W += s
        total_ops += 1

        if is_distinguished(W):
            key = W if W is not None else "inf"
            if key in tame_traps:
                # Collision! T started at hi*G + d_T steps = (hi + d_T)*G
                # W started at k*G + d_W steps = (k + d_W)*G
                # If they meet: hi + tame_traps[key] = k + d_W
                # => k = hi + tame_traps[key] - d_W
                k_candidate = (hi + tame_traps[key] - d_W) % n
                if ec.multiply(G, k_candidate) == Q:
                    return k_candidate, total_ops
                # Try without mod reduction
                k_raw = hi + tame_traps[key] - d_W
                if k_raw >= 0 and k_raw != k_candidate:
                    if ec.multiply(G, k_raw) == Q:
                        return k_raw, total_ops
                # Try nearby values (off-by-one from dp alignment)
                for delta in range(-3, 4):
                    k_try = (k_candidate + delta) % n
                    if ec.multiply(G, k_try) == Q:
                        return k_try, total_ops
            # Store wild DP too for potential wild-wild collisions
            # (not standard but helps on small curves)

        # Early termination: if d_W exceeds the interval + tame distance,
        # the wild kangaroo has overshot
        max_d_T = max(tame_traps.values()) if tame_traps else 0
        if d_W > interval + max_d_T + mean_step * 10:
            break

    return None, total_ops


# ================================================================
# BENCHMARKING
# ================================================================

def run_benchmark():
    """Benchmark all three algorithms across multiple curve sizes."""

    print()
    print("=" * 78)
    print("  POLLARD'S RHO & KANGAROO -- FIXED IMPLEMENTATIONS")
    print("  Benchmark against BSGS baseline on y^2 = x^3 + 7")
    print("=" * 78)

    primes = [101, 503, 1009, 3001, 7919, 15013, 30011, 50021, 100003]
    n_trials = 10

    csv_rows = []

    for p_val in primes:
        ec = SmallEC(p_val, 0, 7)
        N = ec.order
        G = ec.generator

        if G is None or N <= 2:
            print(f"\n  p={p_val}: degenerate curve (|E| = {N}), skipping")
            continue

        group_ord = _subgroup_order(ec, G)
        facts = _factor(N)
        largest_factor = max(f[0] for f in facts) if facts else N

        print(f"\n{'='*78}")
        print(f"  CURVE: y^2 = x^3 + 7 over F_{p_val}")
        print(f"  |E| = {N}, generator order = {group_ord}, largest prime factor = {largest_factor}")
        print(f"  Generator = {G}")

        # Collect results per method
        method_results = {
            "bsgs": {"ops": [], "times": [], "successes": 0},
            "pollard_rho": {"ops": [], "times": [], "successes": 0},
            "kangaroo": {"ops": [], "times": [], "successes": 0},
        }

        print(f"\n  Running {n_trials} trials per method...")
        print(f"  {'Trial':>5s}  {'k':>8s}  {'BSGS':>12s}  {'Rho':>12s}  {'Kangaroo':>12s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")

        for trial in range(n_trials):
            k_target = secrets.randbelow(group_ord - 1) + 1
            Q = ec.multiply(G, k_target)

            trial_results = {}

            # --- BSGS ---
            t0 = time.time()
            k_found, ops = baby_step_giant_step(ec, G, Q, group_order=group_ord)
            dt = (time.time() - t0) * 1000
            correct = k_found is not None and ec.multiply(G, k_found) == Q
            method_results["bsgs"]["ops"].append(ops)
            method_results["bsgs"]["times"].append(dt)
            if correct:
                method_results["bsgs"]["successes"] += 1
            trial_results["bsgs"] = (ops, correct)

            # --- Pollard's rho ---
            t0 = time.time()
            k_found, ops = pollard_rho(ec, G, Q, group_order=group_ord)
            dt = (time.time() - t0) * 1000
            correct = k_found is not None and ec.multiply(G, k_found) == Q
            method_results["pollard_rho"]["ops"].append(ops)
            method_results["pollard_rho"]["times"].append(dt)
            if correct:
                method_results["pollard_rho"]["successes"] += 1
            trial_results["rho"] = (ops, correct)

            # --- Kangaroo (with interval [0, group_ord-1]) ---
            # Give kangaroo a meaningful interval: [k_target - interval/2, k_target + interval/2]
            # In practice we wouldn't know k_target, but for benchmarking
            # we test with a known-bounded range.
            interval_half = int(math.isqrt(group_ord)) * 5
            lo = max(0, k_target - interval_half)
            hi = min(group_ord - 1, k_target + interval_half)
            t0 = time.time()
            k_found, ops = pollard_kangaroo(ec, G, Q, lo, hi, group_order=group_ord)
            dt = (time.time() - t0) * 1000
            correct = k_found is not None and ec.multiply(G, k_found) == Q
            method_results["kangaroo"]["ops"].append(ops)
            method_results["kangaroo"]["times"].append(dt)
            if correct:
                method_results["kangaroo"]["successes"] += 1
            trial_results["kangaroo"] = (ops, correct)

            # Print trial summary
            bsgs_str = f"{trial_results['bsgs'][0]:>6d} {'OK' if trial_results['bsgs'][1] else 'FAIL'}"
            rho_str = f"{trial_results['rho'][0]:>6d} {'OK' if trial_results['rho'][1] else 'FAIL'}"
            kang_str = f"{trial_results['kangaroo'][0]:>6d} {'OK' if trial_results['kangaroo'][1] else 'FAIL'}"
            print(f"  {trial+1:>5d}  {k_target:>8d}  {bsgs_str:>12s}  {rho_str:>12s}  {kang_str:>12s}")

        # Summary for this curve
        print(f"\n  Summary for p={p_val} (|E|={N}, generator order={group_ord}):")
        print(f"  {'Method':<15s} {'Ops Mean':>10s} {'Ops Std':>10s} {'Time Mean':>12s} {'Success':>10s}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

        for method_name in ["bsgs", "pollard_rho", "kangaroo"]:
            r = method_results[method_name]
            ops_arr = np.array(r["ops"], dtype=float)
            times_arr = np.array(r["times"], dtype=float)
            success_rate = r["successes"] / n_trials

            ops_mean = np.mean(ops_arr)
            ops_std = np.std(ops_arr)
            time_mean = np.mean(times_arr)

            print(f"  {method_name:<15s} {ops_mean:>10.1f} {ops_std:>10.1f} "
                  f"{time_mean:>10.2f}ms {success_rate:>9.0%}")

            csv_rows.append({
                "prime": p_val,
                "order": group_ord,
                "method": method_name,
                "ops_mean": f"{ops_mean:.1f}",
                "ops_std": f"{ops_std:.1f}",
                "time_mean_ms": f"{time_mean:.2f}",
                "success_rate": f"{success_rate:.2f}",
            })

    return csv_rows


# ================================================================
# SCALING ANALYSIS
# ================================================================

def scaling_analysis(csv_rows):
    """Fit operations to O(sqrt(N)) and extrapolate to secp256k1."""

    print(f"\n\n{'='*78}")
    print(f"  SCALING ANALYSIS")
    print(f"{'='*78}")

    # Extract data for fitting
    methods = ["bsgs", "pollard_rho", "kangaroo"]

    for method in methods:
        rows = [r for r in csv_rows if r["method"] == method]
        if len(rows) < 3:
            continue

        orders = np.array([int(r["order"]) for r in rows], dtype=float)
        ops = np.array([float(r["ops_mean"]) for r in rows], dtype=float)
        success_rates = np.array([float(r["success_rate"]) for r in rows], dtype=float)

        # Only fit on successful runs
        mask = success_rates > 0.5
        if np.sum(mask) < 3:
            print(f"\n  {method}: insufficient successful data for fitting")
            continue

        orders_fit = orders[mask]
        ops_fit = ops[mask]

        # Fit: ops = c * N^alpha
        # log(ops) = log(c) + alpha * log(N)
        log_orders = np.log2(orders_fit)
        log_ops = np.log2(ops_fit)

        # Linear regression in log-log space
        coeffs = np.polyfit(log_orders, log_ops, 1)
        alpha = coeffs[0]
        log_c = coeffs[1]
        c = 2 ** log_c

        # R^2
        predicted = coeffs[0] * log_orders + coeffs[1]
        ss_res = np.sum((log_ops - predicted) ** 2)
        ss_tot = np.sum((log_ops - np.mean(log_ops)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"\n  {method}:")
        print(f"    Fit: ops = {c:.2f} * N^{alpha:.3f}")
        print(f"    R^2 = {r_squared:.4f}")
        print(f"    Expected exponent: 0.500 (sqrt(N))")
        print(f"    Measured exponent: {alpha:.3f}")

        # Extrapolate to secp256k1
        secp256k1_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        log2_order = math.log2(secp256k1_order)
        estimated_log2_ops = coeffs[0] * log2_order + coeffs[1]
        print(f"    Extrapolation to secp256k1 (N ~ 2^256):")
        print(f"      Estimated ops: 2^{estimated_log2_ops:.1f}")
        if method == "kangaroo":
            print(f"      (kangaroo benchmarked on bounded interval ~ 10*sqrt(N),")
            print(f"       full-range would need 2^128 operations)")

    # Summary
    print(f"\n\n{'='*78}")
    print(f"  EXTRAPOLATION TO secp256k1")
    print(f"{'='*78}")

    print(f"""
  secp256k1 group order: ~2^256

  All three algorithms scale as O(sqrt(N)):

  Algorithm       Time Complexity     Space Complexity    secp256k1 Ops
  --------------- ------------------- ------------------- ----------------
  BSGS            O(sqrt(N))          O(sqrt(N))          ~2^128 (needs 2^128 * 32B = 10^30 TB RAM)
  Pollard rho     O(sqrt(N))          O(1)                ~2^128 (feasible memory, infeasible time)
  Kangaroo        O(sqrt(interval))   O(sqrt(interval))   ~2^128 (full range) / sqrt(interval) if bounded

  At 10^18 group ops/sec (all supercomputers):
    2^128 / 10^18 = 3.4 * 10^20 seconds = 1.1 * 10^13 years

  Bottom line:
  - All classical algorithms hit the sqrt(N) = 2^128 wall
  - This is a PROVEN lower bound (generic group model, Shoup 1997)
  - No classical speedup is possible beyond O(sqrt(N)) for prime-order EC groups
  - Only quantum (Shor's algorithm) achieves polynomial: O(n^3) for n-bit key
""")


# ================================================================
# CORRECTNESS SELF-TEST
# ================================================================

def self_test():
    """Verify all three algorithms on a small curve before benchmarking."""

    print(f"\n  SELF-TEST: verifying all algorithms on small curve...")

    ec = SmallEC(101, 0, 7)
    N = ec.order
    G = ec.generator
    group_ord = _subgroup_order(ec, G)

    print(f"  Curve: y^2 = x^3 + 7 / F_101, |E| = {N}, gen order = {group_ord}, G = {G}")

    n_pass = 0
    n_total = 0

    # Test every possible key
    test_keys = list(range(1, min(group_ord, 50)))

    for k_target in test_keys:
        Q = ec.multiply(G, k_target)

        # BSGS
        k_bsgs, _ = baby_step_giant_step(ec, G, Q, group_order=group_ord)
        bsgs_ok = k_bsgs is not None and ec.multiply(G, k_bsgs) == Q
        n_total += 1
        if bsgs_ok:
            n_pass += 1
        else:
            print(f"    BSGS FAIL: k={k_target}, found={k_bsgs}")

        # Pollard rho
        k_rho, _ = pollard_rho(ec, G, Q, group_order=group_ord)
        rho_ok = k_rho is not None and ec.multiply(G, k_rho) == Q
        n_total += 1
        if rho_ok:
            n_pass += 1
        else:
            print(f"    Rho FAIL: k={k_target}, found={k_rho}")

        # Kangaroo
        lo = max(0, k_target - 20)
        hi = min(group_ord - 1, k_target + 20)
        k_kang, _ = pollard_kangaroo(ec, G, Q, lo, hi, group_order=group_ord)
        kang_ok = k_kang is not None and ec.multiply(G, k_kang) == Q
        n_total += 1
        if kang_ok:
            n_pass += 1
        else:
            print(f"    Kangaroo FAIL: k={k_target}, found={k_kang}")

    print(f"  Self-test: {n_pass}/{n_total} passed")
    if n_pass < n_total:
        fail_rate = (n_total - n_pass) / n_total * 100
        print(f"  WARNING: {fail_rate:.0f}% failure rate in self-test")
    else:
        print(f"  All tests passed.")
    print()
    return n_pass == n_total


# ================================================================
# MAIN
# ================================================================

def main():
    print()
    print("=" * 78)
    print("  POLLARD'S RHO & KANGAROO -- FIXED, BENCHMARKED IMPLEMENTATIONS")
    print("  Corrected from dlp_algorithm_battery.py (which failed on all curves)")
    print("=" * 78)

    # Run self-test first
    self_test()

    # Benchmark
    csv_rows = run_benchmark()

    # Scaling analysis
    scaling_analysis(csv_rows)

    # Write CSV
    csv_path = os.path.expanduser("~/Desktop/pollard_benchmark.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["prime", "order", "method", "ops_mean", "ops_std",
                          "time_mean_ms", "success_rate"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n  Results written to {csv_path}")
    else:
        print(f"\n  No results to write.")

    print("=" * 78)


if __name__ == "__main__":
    main()
