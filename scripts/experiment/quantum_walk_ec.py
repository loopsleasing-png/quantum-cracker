"""Quantum Walk on Elliptic Curve Groups -- The Unexplored Experiment.

Nobody has published this comparison:
  1. Quantum walk on an EC group (geometry-aware)
  2. Same walk on a plain cycle of the same size (control)
  3. Does the curve geometry create faster convergence?

We test multiple coin operators:
  - Hadamard (generic, no geometry)
  - Grover (generic, no geometry)
  - Geometry coin (phase depends on EC point coordinates)
  - Harmonic coin (SH decomposition of point positions on sphere)
  - Curvature coin (uses the curve equation residual)

If EC walk hits the target faster than plain cycle walk with ANY coin,
that's a signal that curve geometry provides computational leverage.

We also map EC points to the harmonic spherical compiler and run
the walk through the SH hierarchy -- testing whether the 78^3 structure
(similar to a welded tree) creates exponential speedup paths.
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
# SMALL ELLIPTIC CURVE IMPLEMENTATION
# ================================================================

class SmallEC:
    """Elliptic curve y^2 = x^3 + ax + b over F_p for small primes."""

    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b
        self.points = self._enumerate_points()
        self.order = len(self.points)
        self.generator = None
        self._point_to_idx = {pt: i for i, pt in enumerate(self.points)}

    def _enumerate_points(self):
        """Find all points on the curve including infinity."""
        points = [None]  # None = point at infinity
        p, a, b = self.p, self.a, self.b

        # Precompute quadratic residues
        qr = {}
        for y in range(p):
            qr[(y * y) % p] = qr.get((y * y) % p, [])
            qr[(y * y) % p].append(y)

        for x in range(p):
            rhs = (x * x * x + a * x + b) % p
            if rhs in qr:
                for y in qr[rhs]:
                    points.append((x, y))

        return points

    def add(self, P, Q):
        """EC point addition."""
        if P is None:
            return Q
        if Q is None:
            return P

        p = self.p
        x1, y1 = P
        x2, y2 = Q

        if x1 == x2 and y1 == (p - y2) % p:
            return None  # P + (-P) = infinity

        if P == Q:
            # Point doubling
            if y1 == 0:
                return None
            inv = pow(2 * y1, p - 2, p)
            lam = (3 * x1 * x1 + self.a) * inv % p
        else:
            if x1 == x2:
                return None
            inv = pow((x2 - x1) % p, p - 2, p)
            lam = (y2 - y1) * inv % p

        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def multiply(self, P, k):
        """Scalar multiplication via double-and-add."""
        if k == 0 or P is None:
            return None
        if k < 0:
            P = self.negate(P)
            k = -k

        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def negate(self, P):
        if P is None:
            return None
        return (P[0], (self.p - P[1]) % self.p)

    def find_generator(self):
        """Find a generator of the full group."""
        for pt in self.points[1:]:  # skip infinity
            if self.multiply(pt, self.order) is None:
                # Check it generates the full group
                is_generator = True
                for d in range(2, int(self.order**0.5) + 1):
                    if self.order % d == 0:
                        if self.multiply(pt, self.order // d) is None:
                            is_generator = False
                            break
                if is_generator:
                    self.generator = pt
                    return pt
        # If no primitive generator, just use first non-trivial point
        self.generator = self.points[1]
        return self.generator

    def point_index(self, P):
        return self._point_to_idx.get(P, -1)

    def discrete_log(self, P, Q):
        """Brute force discrete log: find k such that kP = Q."""
        R = None
        for k in range(self.order + 1):
            if R == Q:
                return k
            R = self.add(R, P)
        return None


# ================================================================
# QUANTUM WALK ENGINE
# ================================================================

class QuantumWalk:
    """Discrete-time quantum walk on a cycle graph (fully vectorized).

    State: |node, coin> where coin in {left, right}
    Total state size: 2 * N amplitudes
    All operations use numpy array ops -- no Python for-loops over nodes.
    """

    def __init__(self, N):
        self.N = N
        self.left = np.zeros(N, dtype=np.complex128)   # coin-left amplitudes
        self.right = np.zeros(N, dtype=np.complex128)   # coin-right amplitudes

    def initialize_at(self, node):
        """Start at given node in equal superposition of coin states."""
        self.left[:] = 0
        self.right[:] = 0
        self.left[node] = 1.0 / np.sqrt(2)
        self.right[node] = 1.0 / np.sqrt(2)

    def step_generic(self, coin_matrix):
        """One walk step with position-independent coin (vectorized)."""
        # Apply coin to all nodes simultaneously
        new_left = coin_matrix[0, 0] * self.left + coin_matrix[0, 1] * self.right
        new_right = coin_matrix[1, 0] * self.left + coin_matrix[1, 1] * self.right

        # Shift: left-coin walkers move left, right-coin walkers move right
        self.left = np.roll(new_left, -1)    # move left (index decreases)
        self.right = np.roll(new_right, 1)   # move right (index increases)

    def step_geometry(self, cos_angles, sin_angles):
        """One walk step with position-dependent coin (vectorized).

        Pre-computed cos/sin arrays avoid recomputing trig each step.
        """
        # Apply position-dependent rotation to all nodes simultaneously
        new_left = cos_angles * self.left + sin_angles * self.right
        new_right = -sin_angles * self.left + cos_angles * self.right

        # Shift
        self.left = np.roll(new_left, -1)
        self.right = np.roll(new_right, 1)

    def probabilities(self):
        """Probability at each node (sum over coin states)."""
        return np.abs(self.left) ** 2 + np.abs(self.right) ** 2

    def prob_at(self, node):
        return float(abs(self.left[node]) ** 2 + abs(self.right[node]) ** 2)


# ================================================================
# COIN OPERATORS
# ================================================================

def hadamard_coin():
    """Standard Hadamard coin -- no geometry awareness."""
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def grover_coin():
    """Grover diffusion coin -- reflects about uniform state."""
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)  # NOT gate (swap L/R)


def dft_coin():
    """DFT coin -- complex phases."""
    omega = np.exp(2j * np.pi / 2)
    return np.array([[1, 1], [1, omega]], dtype=np.complex128) / np.sqrt(2)


def geometry_coin_angles_ec(curve, group_order_list):
    """Build position-dependent coin angles from EC point coordinates.

    Each node's coin angle depends on its EC point's (x, y) coordinates.
    This encodes the curve geometry into the walk dynamics.
    """
    N = len(group_order_list)
    angles = np.zeros(N)
    p = curve.p

    for i, pt in enumerate(group_order_list):
        if pt is None:
            angles[i] = np.pi / 4  # default for infinity
        else:
            x, y = pt
            # Angle derived from point position on the curve
            # Normalize coordinates to [0, 1] and use as rotation
            angles[i] = np.pi * ((x * 0.618033988 + y * 0.381966) % p) / p

    return angles


def curvature_coin_angles(curve, group_order_list):
    """Coin angles based on local curvature of the EC.

    For y^2 = x^3 + ax + b, the curvature at point (x,y) involves
    the second derivative of the curve. Higher curvature = more rotation.
    """
    N = len(group_order_list)
    angles = np.zeros(N)
    p = curve.p

    for i, pt in enumerate(group_order_list):
        if pt is None:
            angles[i] = np.pi / 4
        else:
            x, y = pt
            if y == 0:
                angles[i] = np.pi / 2  # vertical tangent = max curvature
            else:
                # dy/dx = (3x^2 + a) / (2y) on the curve
                # d2y/dx2 involves higher terms
                # Simplified curvature proxy: (3x^2 + a) / (2y) normalized
                dydx_num = (3 * x * x + curve.a) % p
                dydx_den = (2 * y) % p
                # Use the ratio as a phase angle
                ratio = (dydx_num * pow(dydx_den, p - 2, p)) % p
                angles[i] = np.pi * ratio / p

    return angles


def harmonic_coin_angles(curve, group_order_list, n_harmonics=8):
    """Coin angles from spherical harmonic decomposition of point positions.

    Map EC points to sphere, decompose in SH basis, use low-order
    coefficients to determine walk dynamics.
    """
    N = len(group_order_list)
    p = curve.p

    # Map each point to spherical coordinates
    thetas = np.zeros(N)
    phis = np.zeros(N)
    for i, pt in enumerate(group_order_list):
        if pt is None:
            thetas[i] = 0
            phis[i] = 0
        else:
            x, y = pt
            thetas[i] = np.pi * x / p
            phis[i] = 2 * np.pi * y / p

    # Evaluate low-order SH at each point's position
    sh_values = np.zeros((N, n_harmonics))
    idx = 0
    l = 0
    while idx < n_harmonics:
        for m in range(-l, l + 1):
            if idx >= n_harmonics:
                break
            sh_values[:, idx] = sph_harm_y(l, m, thetas, phis).real
            idx += 1
        l += 1

    # Coin angle = weighted sum of SH values
    # Use first few harmonics as the "resonance frequencies"
    weights = np.array([1.0 / (k + 1) for k in range(n_harmonics)])
    angles = np.pi * np.tanh(sh_values @ weights)  # squash to [-pi, pi]

    return angles


def combo_coin_angles(curve, group_order_list):
    """Combine all geometric information into one coin.

    Uses: coordinates, curvature, curve equation residual, and harmonics.
    """
    N = len(group_order_list)
    p = curve.p
    angles = np.zeros(N)

    for i, pt in enumerate(group_order_list):
        if pt is None:
            angles[i] = np.pi / 4
            continue

        x, y = pt

        # Component 1: coordinate-based phase
        coord_phase = 2 * np.pi * (x + y * 1.618033988) / p

        # Component 2: curve equation residual (should be 0 on curve, but mod p gives info)
        # For points ON the curve: y^2 - x^3 - ax - b = 0 mod p
        # Use the "raw" value before mod as extra info
        raw_residual = y * y - x * x * x - curve.a * x - curve.b
        residual_phase = np.pi * (raw_residual % (p * p)) / (p * p)

        # Component 3: discrete log of x-coordinate (mod small prime)
        small_phase = 2 * np.pi * (x % 17) / 17

        # Component 4: bit pattern of coordinates
        xor_bits = bin(x ^ y).count('1')
        bit_phase = np.pi * xor_bits / 256

        # Combine with golden ratio weights
        angles[i] = (coord_phase * 0.4 + residual_phase * 0.3 +
                     small_phase * 0.2 + bit_phase * 0.1) % (2 * np.pi)

    return angles


# ================================================================
# EXPERIMENT RUNNER
# ================================================================

def run_walk_experiment(N, coin_type, coin_param, start_node, target_node,
                        max_steps, label=""):
    """Run a quantum walk and track probability of hitting target.

    coin_type: "matrix" for generic coin, "angles" for geometry-aware
    coin_param: 2x2 matrix for "matrix", angle array for "angles"

    Returns: dict with full walk data
    """
    walk = QuantumWalk(N)
    walk.initialize_at(start_node)

    # Pre-compute cos/sin for geometry coins
    cos_a = sin_a = None
    if coin_type == "angles":
        cos_a = np.cos(coin_param)
        sin_a = np.sin(coin_param)

    prob_history = [walk.prob_at(target_node)]
    max_prob = prob_history[0]
    max_prob_step = 0

    for step in range(1, max_steps + 1):
        if coin_type == "matrix":
            walk.step_generic(coin_param)
        elif coin_type == "angles":
            walk.step_geometry(cos_a, sin_a)

        p_target = walk.prob_at(target_node)
        prob_history.append(p_target)

        if p_target > max_prob:
            max_prob = p_target
            max_prob_step = step

    return {
        "label": label,
        "N": N,
        "max_steps": max_steps,
        "prob_history": np.array(prob_history),
        "max_prob": max_prob,
        "max_prob_step": max_prob_step,
        "final_prob": prob_history[-1],
        "mean_prob": np.mean(prob_history),
    }


# ================================================================
# MAIN EXPERIMENT
# ================================================================

def main():
    print()
    print("=" * 78)
    print("  QUANTUM WALK ON ELLIPTIC CURVES -- The Unexplored Experiment")
    print("  EC geometry-aware walk vs plain cycle walk (control)")
    print("=" * 78)

    # ================================================================
    # BUILD SMALL EC CURVES
    # ================================================================
    # Use Bitcoin's curve equation y^2 = x^3 + 7 over small primes
    # Also test other curve parameters

    curve_configs = [
        (101, 0, 7, "secp-101"),       # ~7-bit
        (211, 0, 7, "secp-211"),       # ~8-bit Bitcoin-like
        (251, 0, 7, "secp-251"),       # ~8-bit Bitcoin-like
        (503, 0, 7, "secp-503"),       # ~9-bit
        (1009, 0, 7, "secp-1009"),     # ~10-bit
        (2003, 0, 7, "secp-2003"),     # ~11-bit
        (4091, 0, 7, "secp-4091"),     # ~12-bit
        (8191, 0, 7, "secp-8191"),     # ~13-bit
        # Alternative curve equations (same field, different geometry)
        (251, 1, 1, "alt-251"),
        (251, 2, 3, "alt2-251"),
        (251, 3, 5, "alt3-251"),
    ]

    print(f"\n  Building elliptic curves...")
    curves = {}

    for p, a, b, name in curve_configs:
        t0 = time.time()
        ec = SmallEC(p, a, b)
        gen = ec.find_generator()
        elapsed = time.time() - t0

        if gen is None:
            print(f"  {name}: p={p}, |E|={ec.order}, no generator found, skipping")
            continue

        curves[name] = ec
        print(f"  {name}: p={p}, y^2=x^3+{a}x+{b}, |E|={ec.order}, "
              f"G={gen}, ({elapsed:.2f}s)")

    # ================================================================
    # BUILD GROUP CYCLE FOR EACH CURVE
    # ================================================================
    print(f"\n  Building group cycles (ordering points by scalar multiplication)...")

    group_data = {}
    for name, ec in curves.items():
        G = ec.generator
        # Build cycle: [0*G, 1*G, 2*G, ..., (order-1)*G]
        cycle = []
        P = None  # start at infinity
        for k in range(ec.order):
            cycle.append(P)
            P = ec.add(P, G)

        # Verify it's a full cycle
        if P is not None:
            # Not a generator of full group -- find subgroup order
            sub_order = len(cycle)
            for i, pt in enumerate(cycle):
                if pt is None and i > 0:
                    sub_order = i
                    break
            cycle = cycle[:sub_order]

        group_data[name] = cycle
        print(f"  {name}: cycle length = {len(cycle)}")

    # ================================================================
    # RUN QUANTUM WALK EXPERIMENTS
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  QUANTUM WALK EXPERIMENTS")
    print(f"  {'='*74}")

    all_results = []

    # Test on curves of increasing size
    test_curves = [n for n in ["secp-101", "secp-211", "secp-251", "secp-503",
                                "secp-1009", "secp-2003", "secp-4091", "secp-8191",
                                "alt-251", "alt2-251", "alt3-251"]
                   if n in curves]

    for curve_name in test_curves:
        ec = curves[curve_name]
        cycle = group_data[curve_name]
        N = len(cycle)

        if N < 10:
            continue

        # Pick a target (private key)
        target_k = N // 3 + 7  # deterministic, non-trivial
        target_k = target_k % N
        start_node = 0  # public key position in cycle
        target_node = target_k

        max_steps = min(N * 4, 5000)  # cap for large curves

        print(f"\n  --- {curve_name}: N={N}, target_k={target_k}, "
              f"max_steps={max_steps} ---")

        # ---- COIN 1: Hadamard on plain cycle (CONTROL) ----
        t0 = time.time()
        result_hadamard_plain = run_walk_experiment(
            N, "matrix", hadamard_coin(), start_node, target_node,
            max_steps, f"{curve_name}/hadamard/plain")
        t1 = time.time()

        print(f"  Hadamard (plain cycle): "
              f"max_P={result_hadamard_plain['max_prob']:.6f} "
              f"at step {result_hadamard_plain['max_prob_step']}  "
              f"({t1-t0:.2f}s)")
        all_results.append(result_hadamard_plain)

        # ---- COIN 2: Hadamard on EC (should be same as plain if geometry doesn't matter) ----
        # (Same walk -- but we'll compare with geometry coins below)

        # ---- COIN 3: Geometry coin on EC ----
        geo_angles = geometry_coin_angles_ec(ec, cycle)
        t0 = time.time()
        result_geo = run_walk_experiment(
            N, "angles", geo_angles, start_node, target_node,
            max_steps, f"{curve_name}/geometry/ec")
        t1 = time.time()

        print(f"  Geometry coin (EC):     "
              f"max_P={result_geo['max_prob']:.6f} "
              f"at step {result_geo['max_prob_step']}  "
              f"({t1-t0:.2f}s)")
        all_results.append(result_geo)

        # ---- COIN 4: Curvature coin ----
        curv_angles = curvature_coin_angles(ec, cycle)
        t0 = time.time()
        result_curv = run_walk_experiment(
            N, "angles", curv_angles, start_node, target_node,
            max_steps, f"{curve_name}/curvature/ec")
        t1 = time.time()

        print(f"  Curvature coin (EC):    "
              f"max_P={result_curv['max_prob']:.6f} "
              f"at step {result_curv['max_prob_step']}  "
              f"({t1-t0:.2f}s)")
        all_results.append(result_curv)

        # ---- COIN 5: Harmonic coin ----
        harm_angles = harmonic_coin_angles(ec, cycle)
        t0 = time.time()
        result_harm = run_walk_experiment(
            N, "angles", harm_angles, start_node, target_node,
            max_steps, f"{curve_name}/harmonic/ec")
        t1 = time.time()

        print(f"  Harmonic coin (EC):     "
              f"max_P={result_harm['max_prob']:.6f} "
              f"at step {result_harm['max_prob_step']}  "
              f"({t1-t0:.2f}s)")
        all_results.append(result_harm)

        # ---- COIN 6: Combo coin (everything) ----
        combo_angles = combo_coin_angles(ec, cycle)
        t0 = time.time()
        result_combo = run_walk_experiment(
            N, "angles", combo_angles, start_node, target_node,
            max_steps, f"{curve_name}/combo/ec")
        t1 = time.time()

        print(f"  Combo coin (EC):        "
              f"max_P={result_combo['max_prob']:.6f} "
              f"at step {result_combo['max_prob_step']}  "
              f"({t1-t0:.2f}s)")
        all_results.append(result_combo)

        # ---- COIN 7: DFT coin (control 2) ----
        t0 = time.time()
        result_dft = run_walk_experiment(
            N, "matrix", dft_coin(), start_node, target_node,
            max_steps, f"{curve_name}/dft/plain")
        t1 = time.time()

        print(f"  DFT coin (plain):       "
              f"max_P={result_dft['max_prob']:.6f} "
              f"at step {result_dft['max_prob_step']}  "
              f"({t1-t0:.2f}s)")
        all_results.append(result_dft)

        # ---- COIN 8: Random angles (noise baseline) ----
        np.random.seed(42)
        random_angles = np.random.uniform(0, 2 * np.pi, N)
        t0 = time.time()
        result_random = run_walk_experiment(
            N, "angles", random_angles, start_node, target_node,
            max_steps, f"{curve_name}/random_angles/noise")
        t1 = time.time()

        print(f"  Random coin (noise):    "
              f"max_P={result_random['max_prob']:.6f} "
              f"at step {result_random['max_prob_step']}  "
              f"({t1-t0:.2f}s)")
        all_results.append(result_random)

        # ---- COMPARISON ----
        baseline = 1.0 / N  # uniform random probability
        print(f"\n  Baseline (uniform): {baseline:.6f}")
        print(f"  Summary for {curve_name}:")
        for r in [result_hadamard_plain, result_geo, result_curv,
                  result_harm, result_combo, result_dft, result_random]:
            improvement = r["max_prob"] / baseline
            label_short = r["label"].split("/")[1]
            print(f"    {label_short:18s}: max_P={r['max_prob']:.6f} "
                  f"({improvement:.1f}x baseline) at step {r['max_prob_step']}")

    # ================================================================
    # CROSS-SIZE ANALYSIS
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  CROSS-SIZE ANALYSIS: Does EC advantage scale?")
    print(f"  {'='*74}")

    print(f"\n  {'Curve':>16s}  {'N':>8s}  {'Hadamard':>10s}  {'Geometry':>10s}  "
          f"{'Curvature':>10s}  {'Harmonic':>10s}  {'Combo':>10s}  {'EC/Plain':>10s}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    scaling_data = []
    for curve_name in test_curves:
        if curve_name not in curves:
            continue

        ec = curves[curve_name]
        cycle = group_data[curve_name]
        N = len(cycle)

        # Find results for this curve
        hadamard_r = None
        geo_r = None
        curv_r = None
        harm_r = None
        combo_r = None

        for r in all_results:
            if r["label"].startswith(curve_name):
                if "hadamard" in r["label"]:
                    hadamard_r = r
                elif "geometry" in r["label"]:
                    geo_r = r
                elif "curvature" in r["label"]:
                    curv_r = r
                elif "harmonic" in r["label"]:
                    harm_r = r
                elif "combo" in r["label"]:
                    combo_r = r

        if not all([hadamard_r, geo_r, curv_r, harm_r, combo_r]):
            continue

        # Best EC coin vs best plain coin
        best_ec = max(geo_r["max_prob"], curv_r["max_prob"],
                      harm_r["max_prob"], combo_r["max_prob"])
        best_plain = hadamard_r["max_prob"]
        ratio = best_ec / best_plain if best_plain > 0 else 0

        print(f"  {curve_name:>16s}  {N:8d}  "
              f"{hadamard_r['max_prob']:10.6f}  "
              f"{geo_r['max_prob']:10.6f}  "
              f"{curv_r['max_prob']:10.6f}  "
              f"{harm_r['max_prob']:10.6f}  "
              f"{combo_r['max_prob']:10.6f}  "
              f"{ratio:10.3f}x")

        scaling_data.append({
            "curve": curve_name,
            "N": N,
            "log2_N": np.log2(N),
            "hadamard_max": hadamard_r["max_prob"],
            "geometry_max": geo_r["max_prob"],
            "curvature_max": curv_r["max_prob"],
            "harmonic_max": harm_r["max_prob"],
            "combo_max": combo_r["max_prob"],
            "ec_plain_ratio": ratio,
        })

    # ================================================================
    # SCALING TREND
    # ================================================================
    if len(scaling_data) >= 3:
        print(f"\n  Scaling trend (does EC advantage grow with size?):")
        ns = [d["N"] for d in scaling_data]
        ratios = [d["ec_plain_ratio"] for d in scaling_data]

        if len(set(ratios)) > 1:
            log_ns = np.log2(ns)
            slope, intercept, r_val, _, _ = stats.linregress(log_ns, ratios)
            print(f"    Linear fit: ratio = {slope:.4f} * log2(N) + {intercept:.4f}")
            print(f"    R^2 = {r_val**2:.6f}")
            if slope > 0.01:
                print(f"    TREND: EC advantage GROWS with size (slope={slope:.4f})")
                print(f"    Extrapolation to 256-bit: ratio ~ {slope * 256 + intercept:.2f}x")
            elif slope < -0.01:
                print(f"    TREND: EC advantage SHRINKS with size")
            else:
                print(f"    TREND: No clear scaling -- EC and plain are equivalent")

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n  {'='*74}")
    print(f"  VERDICT")
    print(f"  {'='*74}")

    if scaling_data:
        best_ratio = max(d["ec_plain_ratio"] for d in scaling_data)
        best_curve = max(scaling_data, key=lambda d: d["ec_plain_ratio"])

        if best_ratio > 2.0:
            print(f"\n  *** SIGNAL: EC geometry gives {best_ratio:.1f}x advantage ***")
            print(f"  Best on {best_curve['curve']} (N={best_curve['N']})")
            print(f"  This warrants deeper investigation.")
        elif best_ratio > 1.2:
            print(f"\n  Marginal signal: EC geometry gives {best_ratio:.2f}x advantage")
            print(f"  Could be noise. Test on more curves and sizes.")
        else:
            print(f"\n  No signal: EC geometry provides no walk advantage ({best_ratio:.3f}x)")
            print(f"  The curve geometry does not create useful interference patterns.")
            print(f"  Quantum walks on EC groups behave the same as on plain cycles.")

    # ================================================================
    # WRITE CSV
    # ================================================================
    csv_path = "/Users/kjm/Desktop/quantum_walk_ec.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "N", "max_prob", "max_prob_step",
                          "final_prob", "mean_prob", "max_steps"])
        for r in all_results:
            writer.writerow([r["label"], r["N"], f"{r['max_prob']:.8f}",
                            r["max_prob_step"], f"{r['final_prob']:.8f}",
                            f"{r['mean_prob']:.8f}", r["max_steps"]])

    if scaling_data:
        scaling_path = "/Users/kjm/Desktop/quantum_walk_scaling.csv"
        with open(scaling_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scaling_data[0].keys())
            writer.writeheader()
            for row in scaling_data:
                writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                                for k, v in row.items()})
        print(f"\n  Scaling: {scaling_path}")

    print(f"  Results: {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
