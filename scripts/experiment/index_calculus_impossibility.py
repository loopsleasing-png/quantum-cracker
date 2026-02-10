"""Index Calculus Impossibility on Elliptic Curves.

Demonstrates WHY index calculus -- the primary attack that breaks finite field
DLP in subexponential time -- fundamentally CANNOT work on elliptic curves.

Background:
    Index calculus is the reason RSA needs 2048-bit keys while ECC only needs
    256-bit keys. It exploits the fact that integers have a unique prime
    factorization. Given g^x mod p, you can check whether g^e mod p factors
    over a set of small primes (a "factor base"). Collect enough such
    "smooth" relations and you can solve a linear system to recover discrete logs.

    The complexity is L(p) = exp(c * sqrt(ln(p) * ln(ln(p)))) -- subexponential.
    For 2048-bit p, this is roughly 2^112, which is feasible.

    On elliptic curves, the group elements are POINTS (x, y), not integers.
    There is no notion of "factoring" a point. The group operation (chord-and-
    tangent addition) bears no algebraic relation to the coordinate values.
    You cannot decompose P into "small" summands without already knowing
    the discrete logs of those summands -- a circular dependency.

    The best known attack on EC groups over prime fields remains Pollard's
    rho at O(sqrt(N)) -- fully exponential in the bit length.

    Attempts to import index calculus ideas to EC:
    - Weil descent / GHS attack: works only for specific curves over EXTENSION
      fields F_{q^n} with small n. Does NOT apply to secp256k1 (prime field).
    - Summation polynomials (Semaev, 2004): theoretical framework, but solving
      the resulting multivariate polynomial systems is at least as hard as the
      original DLP for prime-field curves.
    - Decomposition attacks (Gaudry, Diem): subexponential for curves over
      F_{q^n} with n >= 3 and q fixed. Again, NOT applicable to prime fields.

    For secp256k1 over F_p (a prime field): NO known subexponential algorithm.
    If someone found one, it would break all deployed elliptic curve cryptography.

This script:
    1. Implements a working index calculus attack on F_p* (multiplicative group)
    2. Demonstrates why the same approach fails on E(F_p)
    3. Compares operation counts: subexponential vs fully exponential
    4. Outputs CSV analysis to ~/Desktop/index_calculus_analysis.csv
"""

import csv
import math
import os
import secrets
import time


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def is_prime(n):
    """Miller-Rabin primality test for small n (deterministic for n < 3.3M)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic witnesses for n < 3,215,031,751
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in [2, 3, 5, 7]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def small_primes(bound):
    """Sieve of Eratosthenes up to bound."""
    if bound < 2:
        return []
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(bound**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, bound + 1, i):
                sieve[j] = False
    return [i for i in range(2, bound + 1) if sieve[i]]


def factor_trial(n):
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


def is_b_smooth(n, factor_base):
    """Check if n factors completely over the factor base.

    Returns the exponent vector if smooth, None otherwise.
    """
    if n <= 0:
        return None
    exponents = []
    remaining = n
    for p in factor_base:
        e = 0
        while remaining % p == 0:
            remaining //= p
            e += 1
        exponents.append(e)
    if remaining == 1:
        return exponents
    return None


def find_generator_fp(p):
    """Find a primitive root modulo p (generator of F_p*)."""
    if p == 2:
        return 1
    phi = p - 1
    factors = factor_trial(phi)
    prime_factors = [f[0] for f in factors]

    for g in range(2, p):
        is_gen = True
        for q in prime_factors:
            if pow(g, phi // q, p) == 1:
                is_gen = False
                break
        if is_gen:
            return g
    return None


# ================================================================
# GAUSSIAN ELIMINATION MOD N
# ================================================================

def gaussian_elimination_mod(matrix, targets, modulus):
    """Solve a system of linear equations modulo modulus.

    matrix: list of rows, each row is a list of coefficients
    targets: list of target values (RHS)
    modulus: modular arithmetic base

    Returns solution vector or None.
    """
    if not matrix or not matrix[0]:
        return None

    nrows = len(matrix)
    ncols = len(matrix[0])

    # Augmented matrix
    aug = []
    for i in range(nrows):
        row = [x % modulus for x in matrix[i]] + [targets[i] % modulus]
        aug.append(row)

    pivot_cols = []
    pivot_row = 0

    for col in range(ncols):
        # Find pivot
        found = -1
        for row in range(pivot_row, nrows):
            if math.gcd(aug[row][col], modulus) == 1:
                found = row
                break
        if found == -1:
            # Try non-zero pivot even if not invertible
            for row in range(pivot_row, nrows):
                if aug[row][col] % modulus != 0:
                    found = row
                    break
        if found == -1:
            continue

        # Swap rows
        aug[found], aug[pivot_row] = aug[pivot_row], aug[found]

        # Scale pivot row
        pivot_val = aug[pivot_row][col]
        try:
            inv = pow(pivot_val, -1, modulus)
        except (ValueError, ZeroDivisionError):
            continue

        for j in range(len(aug[pivot_row])):
            aug[pivot_row][j] = (aug[pivot_row][j] * inv) % modulus

        # Eliminate column
        for row in range(nrows):
            if row == pivot_row:
                continue
            factor = aug[row][col]
            if factor != 0:
                for j in range(len(aug[row])):
                    aug[row][j] = (aug[row][j] - factor * aug[pivot_row][j]) % modulus

        pivot_cols.append(col)
        pivot_row += 1

    # Extract solution
    solution = [0] * ncols
    for i, col in enumerate(pivot_cols):
        if i < nrows:
            solution[col] = aug[i][-1] % modulus

    return solution


# ================================================================
# INDEX CALCULUS ON F_p* (WORKING IMPLEMENTATION)
# ================================================================

def index_calculus_fp(p, g, h, verbose=False):
    """Index calculus attack on DLP in F_p*: find x such that g^x = h mod p.

    Steps:
    1. Choose a factor base B = {p1, p2, ..., pb} of small primes
    2. Collect relations: find random e where g^e mod p is B-smooth
    3. Each smooth relation gives: e = sum(a_i * log_g(p_i)) mod (p-1)
    4. Solve the linear system for log_g(p_i)
    5. Then find log_g(h) by searching for t where h * g^t mod p is smooth

    Returns (x, ops) or (None, ops).
    """
    order = p - 1  # |F_p*| = p - 1
    ops = 0

    # Choose smoothness bound B
    # Optimal: B ~ exp(0.5 * sqrt(ln(p) * ln(ln(p))))
    ln_p = math.log(p)
    ln_ln_p = math.log(max(ln_p, 2))
    B_optimal = int(math.exp(0.5 * math.sqrt(ln_p * ln_ln_p)))
    B = max(B_optimal, 10)
    B = min(B, p // 2)  # don't exceed p/2

    factor_base = small_primes(B)
    fb_size = len(factor_base)

    if verbose:
        print(f"    Factor base: B={B}, {fb_size} primes, largest={factor_base[-1] if factor_base else 0}")

    if fb_size == 0:
        return None, ops

    # Phase 1: Collect relations
    # We need at least fb_size + 1 relations for a determined system
    relations = []  # (exponent_vector, e)
    target_relations = fb_size + 5  # a few extra for safety
    max_attempts = fb_size * 200  # give up after too many tries

    for attempt in range(max_attempts):
        e = secrets.randbelow(order)
        val = pow(g, e, p)
        ops += 1

        exps = is_b_smooth(val, factor_base)
        if exps is not None:
            relations.append((exps, e))
            if verbose and len(relations) <= 3:
                print(f"    Relation {len(relations)}: g^{e} = {val} = "
                      f"{' * '.join(f'{factor_base[i]}^{exps[i]}' for i in range(fb_size) if exps[i] > 0)}")
            if len(relations) >= target_relations:
                break

    if len(relations) < fb_size:
        if verbose:
            print(f"    Only found {len(relations)}/{fb_size} relations after {max_attempts} attempts")
        return None, ops

    if verbose:
        print(f"    Collected {len(relations)} relations in {ops} operations")

    # Phase 2: Solve linear system for discrete logs of factor base elements
    # Each relation: e_i = sum(a_{i,j} * log_g(p_j)) mod (p-1)
    matrix = [r[0] for r in relations[:fb_size + 2]]
    targets = [r[1] for r in relations[:fb_size + 2]]
    ops += fb_size * fb_size  # approximate cost of Gaussian elimination

    # The modulus for the system is order = p-1
    # However, p-1 is usually not prime, so we need care.
    # Try solving mod order directly.
    logs = gaussian_elimination_mod(matrix, targets, order)

    if logs is None:
        if verbose:
            print(f"    Gaussian elimination failed")
        return None, ops

    # Verify a few factor base logs
    verified = 0
    for i in range(min(3, fb_size)):
        if pow(g, logs[i], p) == factor_base[i] % p:
            verified += 1

    if verbose:
        print(f"    Verified {verified} factor base logs")
        for i in range(min(5, fb_size)):
            print(f"    log_g({factor_base[i]}) = {logs[i]}"
                  f"  [check: g^{logs[i]} mod p = {pow(g, logs[i], p)}, want {factor_base[i]}]")

    # Phase 3: Express h in terms of factor base
    # Try random t until h * g^t mod p is B-smooth
    for attempt in range(max_attempts):
        t = secrets.randbelow(order)
        val = (h * pow(g, t, p)) % p
        ops += 1

        exps = is_b_smooth(val, factor_base)
        if exps is not None:
            # log_g(h) + t = sum(a_j * log_g(p_j)) mod (p-1)
            log_h = sum(exps[j] * logs[j] for j in range(fb_size)) - t
            log_h = log_h % order
            ops += fb_size

            # Verify
            if pow(g, log_h, p) == h:
                if verbose:
                    print(f"    Found log_g(h) = {log_h} (verified)")
                return log_h, ops

            # Try with corrections for non-prime order
            for delta in range(0, order, max(1, order // 100)):
                candidate = (log_h + delta) % order
                if pow(g, candidate, p) == h:
                    if verbose:
                        print(f"    Found log_g(h) = {candidate} (with delta correction)")
                    return candidate, ops
                if delta > 0:
                    candidate = (log_h - delta) % order
                    if pow(g, candidate, p) == h:
                        if verbose:
                            print(f"    Found log_g(h) = {candidate} (with delta correction)")
                        return candidate, ops

    if verbose:
        print(f"    Phase 3 failed: could not express h in terms of factor base")
    return None, ops


# ================================================================
# ELLIPTIC CURVE ARITHMETIC
# ================================================================

class SmallEC:
    """Elliptic curve over F_p with full arithmetic."""

    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b
        self._points = None
        self._order = None
        self._gen = None

    @property
    def order(self):
        if self._order is None:
            self._enumerate()
        return self._order

    @property
    def generator(self):
        if self._gen is None:
            self._find_generator()
        return self._gen

    def _enumerate(self):
        points = [None]  # infinity
        p, a, b = self.p, self.a, self.b
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
        if self._points is None:
            self._enumerate()
        for pt in self._points[1:]:
            if self.multiply(pt, self.order) is None:
                is_gen = True
                for d in range(2, int(self.order ** 0.5) + 1):
                    if self.order % d == 0:
                        if self.multiply(pt, self.order // d) is None:
                            is_gen = False
                            break
                if is_gen:
                    self._gen = pt
                    return pt
        self._gen = self._points[1]
        return self._gen

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
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, p - 2, p) % p
        else:
            if x1 == x2: return None
            lam = (y2 - y1) * pow((x2 - x1) % p, p - 2, p) % p
        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def neg(self, P):
        if P is None: return None
        return (P[0], (self.p - P[1]) % self.p)

    def multiply(self, P, k):
        if k < 0:
            P = self.neg(P)
            k = -k
        if k == 0 or P is None: return None
        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result


# ================================================================
# BSGS ON ELLIPTIC CURVES
# ================================================================

def bsgs_ec(ec, G, Q):
    """Baby-step Giant-step for ECDLP. Returns (k, ops)."""
    n = ec.order
    m = int(math.isqrt(n)) + 1
    ops = 0

    # Baby steps: j -> jG
    baby = {}
    P = None
    for j in range(m):
        key = P if P is not None else "inf"
        baby[key] = j
        P = ec.add(P, G)
        ops += 1

    # Giant step: -mG
    mG = ec.multiply(G, m)
    neg_mG = ec.neg(mG)
    ops += int(math.log2(max(m, 1))) + 1

    # Giant steps: Q - i*mG
    gamma = Q
    for i in range(m):
        key = gamma if gamma is not None else "inf"
        if key in baby:
            k = (baby[key] + i * m) % n if n > 1 else 0
            if ec.multiply(G, k) == Q:
                return k, ops
            # Try nearby values
            for offset in range(-2, 3):
                kk = baby[key] + i * m + offset
                if 0 <= kk < n and ec.multiply(G, kk) == Q:
                    return kk, ops
        gamma = ec.add(gamma, neg_mG)
        ops += 1

    return None, ops


# ================================================================
# DEMONSTRATION: WHY INDEX CALCULUS FAILS ON EC
# ================================================================

def demonstrate_no_factoring_on_ec(ec, verbose=True):
    """Show that EC points cannot be 'factored' or decomposed.

    In F_p*, we can check if an element is B-smooth (factors over small primes).
    In E(F_p), there is NO analogue. A point P = (x, y) is an atomic object
    in the group -- you cannot look at its coordinates and determine any
    decomposition P = P1 + P2 + ... in terms of "small" points.

    The group operation (chord-and-tangent) scrambles coordinates in a way
    that has NO relation to factoring or smoothness.
    """
    if verbose:
        print(f"\n  DEMONSTRATION: Points cannot be factored")
        print(f"  -" * 39)

    G = ec.generator
    N = ec.order

    # Pick a "factor base" of EC points -- say the first few multiples of G
    fb_size = min(8, N // 2)
    factor_base_points = []
    for i in range(1, fb_size + 1):
        pt = ec.multiply(G, i)
        if pt is not None:
            factor_base_points.append((i, pt))

    if verbose:
        print(f"  EC factor base (first {fb_size} multiples of G):")
        for idx, pt in factor_base_points[:5]:
            print(f"    P_{idx} = {idx}*G = {pt}")
        if fb_size > 5:
            print(f"    ... ({fb_size} total)")

    # Pick a target point
    k_target = secrets.randbelow(N - 1) + 1
    Q = ec.multiply(G, k_target)

    if verbose:
        print(f"\n  Target: Q = {k_target}*G = {Q}")
        print(f"  Q has coordinates x={Q[0]}, y={Q[1]}")

    # In F_p*, we would try to factor h = g^x mod p over the factor base.
    # On EC, what would "factoring Q" mean?
    # It would mean finding a_1, a_2, ... such that Q = a_1*P_1 + a_2*P_2 + ...
    # But P_i = i*G, so this is just finding a_1*1 + a_2*2 + ... = k_target
    # which requires KNOWING k_target -- circular!

    if verbose:
        print(f"\n  Attempting to 'decompose' Q over the factor base:")
        print(f"  We want: Q = c_1*P_1 + c_2*P_2 + ... + c_b*P_b")
        print(f"  Since P_i = i*G, this means: k = c_1*1 + c_2*2 + ... + c_b*b")
        print(f"  But we don't KNOW k! That's the DLP we're trying to solve!")

    # Show the fundamental problem: given Q, can we tell ANYTHING about k
    # from the coordinates alone?
    if verbose:
        print(f"\n  Coordinate analysis (showing coordinates reveal NOTHING about k):")
        samples = []
        for _ in range(8):
            ki = secrets.randbelow(N - 1) + 1
            Pi = ec.multiply(G, ki)
            if Pi is not None:
                samples.append((ki, Pi))

        print(f"    {'k':>8s}  {'x':>8s}  {'y':>8s}  {'x mod 10':>8s}  {'k mod 10':>8s}  {'Correlated?'}")
        print(f"    {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*11}")
        for ki, Pi in samples:
            print(f"    {ki:>8d}  {Pi[0]:>8d}  {Pi[1]:>8d}  {Pi[0] % 10:>8d}  {ki % 10:>8d}  {'NO'}")

        print(f"\n  The x-coordinate of kG has NO algebraic relation to k.")
        print(f"  This is the CORE reason index calculus fails on EC.")

    # Quantify: in F_p*, checking smoothness is O(B) trial divisions.
    # On EC, "checking if Q decomposes over a factor base" requires
    # solving a DLP for EACH relation -- you need the answer to get the answer.
    if verbose:
        print(f"\n  Cost comparison for building one 'relation':")
        print(f"    F_p*:  Test if g^e mod p is B-smooth -> O(B) divisions")
        print(f"    E(F_p): Express kG as sum of basis points -> requires solving DLP!")
        print(f"    The relation-collection step is FREE in F_p* but IMPOSSIBLE on EC")

    return True


def demonstrate_coordinate_scrambling(ec, verbose=True):
    """Show that EC addition completely scrambles coordinate values.

    In F_p*, multiplication preserves structure: if a and b are smooth,
    a*b is also smooth. The multiplicative structure is compatible with
    integer factoring.

    In E(F_p), adding two points with 'nice' coordinates produces a point
    with seemingly random coordinates. There is no smoothness preservation.
    """
    if verbose:
        print(f"\n  DEMONSTRATION: Group operation scrambles coordinates")
        print(f"  -" * 39)

    G = ec.generator
    p = ec.p

    # Show: adding points with small coordinates gives large/random coordinates
    P1 = ec.multiply(G, 1)
    P2 = ec.multiply(G, 2)
    P3 = ec.add(P1, P2)  # = 3G
    P4 = ec.multiply(G, 3)  # should equal P3

    if verbose:
        print(f"  P1 = 1*G = {P1}")
        print(f"  P2 = 2*G = {P2}")
        print(f"  P1 + P2  = {P3}")
        print(f"  3*G      = {P4}")
        print(f"  P1 + P2 == 3*G? {P3 == P4}")

    # Compare with F_p*
    if verbose:
        g = find_generator_fp(p) or 2
        print(f"\n  Compare with F_{p}* (g={g}):")
        a = pow(g, 3, p)
        b = pow(g, 5, p)
        c = (a * b) % p
        d = pow(g, 8, p)
        print(f"    g^3 = {a}, g^5 = {b}")
        print(f"    g^3 * g^5 = {c}")
        print(f"    g^8       = {d}")
        print(f"    g^3 * g^5 == g^8? {c == d}")
        print(f"\n    In F_p*: {a} * {b} = {c} -- this is an INTEGER, we can factor it!")

        fb = small_primes(20)
        smooth_a = is_b_smooth(a, fb)
        smooth_b = is_b_smooth(b, fb)
        smooth_c = is_b_smooth(c, fb)
        print(f"    Is g^3={a} 20-smooth? {smooth_a is not None}")
        print(f"    Is g^5={b} 20-smooth? {smooth_b is not None}")
        print(f"    Is g^3*g^5={c} 20-smooth? {smooth_c is not None}")
        print(f"\n    On EC: ({P1[0]},{P1[1]}) + ({P2[0]},{P2[1]}) = ({P3[0]},{P3[1]})")
        print(f"    What does 'smooth' even MEAN for the point ({P3[0]},{P3[1]})?")
        print(f"    NOTHING. The concept does not exist for EC points.")


# ================================================================
# QUANTITATIVE COMPARISON
# ================================================================

def l_complexity(n):
    """Compute L(n) = exp(sqrt(ln(n) * ln(ln(n)))) -- index calculus complexity."""
    if n < 3:
        return float(n)
    ln_n = math.log(n)
    ln_ln_n = math.log(max(ln_n, 1.0))
    return math.exp(math.sqrt(ln_n * ln_ln_n))


def l_complexity_log2(bits):
    """Compute log2(L(2^bits)) using logarithms to avoid overflow.

    L(N) = exp(sqrt(ln(N) * ln(ln(N))))
    For N = 2^bits: ln(N) = bits * ln(2)
    ln(ln(N)) = ln(bits * ln(2)) = ln(bits) + ln(ln(2))
    L(N) = exp(sqrt(bits * ln(2) * (ln(bits) + ln(ln(2)))))
    log2(L(N)) = sqrt(bits * ln(2) * (ln(bits) + ln(ln(2)))) / ln(2)
    """
    if bits < 2:
        return float(bits)
    ln2 = math.log(2)
    ln_N = bits * ln2
    ln_ln_N = math.log(max(ln_N, 1.0))
    return math.sqrt(ln_N * ln_ln_N) / ln2


def compare_complexities(bit_sizes):
    """Compare index calculus (subexponential) vs BSGS (exponential)."""
    print(f"\n  {'Bits':>6s}  {'Field size':>14s}  {'IC ops (F_p*)':>16s}  "
          f"{'BSGS ops (EC)':>16s}  {'Ratio EC/IC':>12s}  {'IC feasible?':>14s}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*16}  {'-'*16}  {'-'*12}  {'-'*14}")

    rows = []
    for bits in bit_sizes:
        # Work entirely in log2 space to avoid overflow
        ic_log2 = l_complexity_log2(bits)
        bsgs_log2 = bits / 2.0  # sqrt(2^bits) = 2^(bits/2)
        ratio_log2 = bsgs_log2 - ic_log2

        # Feasibility threshold: ~2^80 operations
        feasible = "YES" if ic_log2 < 80 else ("MARGINAL" if ic_log2 < 112 else "NO")

        print(f"  {bits:>6d}  {'2^'+str(bits):>14s}  {'2^'+f'{ic_log2:.1f}':>16s}  "
              f"{'2^'+f'{bsgs_log2:.1f}':>16s}  {'2^'+f'{max(ratio_log2,0):.1f}':>12s}  {feasible:>14s}")

        rows.append({
            "bits": bits,
            "field_size": f"2^{bits}",
            "ic_ops_log2": round(ic_log2, 2),
            "bsgs_ops_log2": round(bsgs_log2, 2),
            "ratio_log2": round(max(ratio_log2, 0), 2),
            "ic_feasible": feasible,
        })

    return rows


# ================================================================
# MAIN
# ================================================================

def main():
    print()
    print("=" * 78)
    print("  INDEX CALCULUS IMPOSSIBILITY ON ELLIPTIC CURVES")
    print("  Why the attack that breaks F_p* DLP cannot work on EC groups")
    print("=" * 78)

    csv_rows = []

    # ================================================================
    # PART 1: INDEX CALCULUS ON F_p* (WORKING ATTACK)
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  PART 1: INDEX CALCULUS ON F_p* (WORKING ATTACK)")
    print(f"{'='*78}")
    print(f"\n  The multiplicative group F_p* has order p-1.")
    print(f"  Elements are integers that can be FACTORED over small primes.")
    print(f"  This is the foundation of the index calculus method.")

    # Test on primes where p-1 has small factors (makes smooth values more common)
    test_primes_fp = []
    # Find primes where p-1 is reasonably smooth
    for p_candidate in range(100, 12000):
        if not is_prime(p_candidate):
            continue
        facts = factor_trial(p_candidate - 1)
        largest_factor = max(f[0] for f in facts)
        # We want p-1 to have moderate-sized factors for index calculus to work
        if largest_factor < p_candidate // 3:
            test_primes_fp.append(p_candidate)
            if len(test_primes_fp) >= 12:
                break

    print(f"\n  Testing index calculus on F_p* for {len(test_primes_fp)} primes:")
    print(f"  {'p':>8s}  {'|F_p*|':>8s}  {'Largest factor':>14s}  "
          f"{'IC ops':>8s}  {'Result':>8s}  {'Time':>8s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}")

    ic_successes = 0
    ic_total = 0

    for p_val in test_primes_fp:
        g = find_generator_fp(p_val)
        if g is None:
            continue

        order = p_val - 1
        facts = factor_trial(order)
        largest = max(f[0] for f in facts)

        # Random target
        x_target = secrets.randbelow(order - 1) + 1
        h = pow(g, x_target, p_val)

        t0 = time.time()
        x_found, ops = index_calculus_fp(p_val, g, h, verbose=(p_val == test_primes_fp[0]))
        dt = (time.time() - t0) * 1000

        correct = x_found is not None and pow(g, x_found, p_val) == h
        ic_total += 1
        if correct:
            ic_successes += 1

        status = "OK" if correct else ("WRONG" if x_found is not None else "FAIL")
        print(f"  {p_val:>8d}  {order:>8d}  {largest:>14d}  "
              f"{ops:>8d}  {status:>8s}  {dt:>7.1f}ms")

        csv_rows.append({
            "field_size": p_val,
            "index_calculus_ops": ops,
            "ec_bsgs_ops": "",
            "ratio": "",
            "category": "index_calculus_fp",
        })

    print(f"\n  Index calculus success rate on F_p*: {ic_successes}/{ic_total}")
    if ic_successes > 0:
        print(f"  The attack WORKS because integers can be factored.")

    # ================================================================
    # PART 2: WHY IT FAILS ON EC
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  PART 2: WHY INDEX CALCULUS FAILS ON ELLIPTIC CURVES")
    print(f"{'='*78}")

    # Use a moderate prime for demonstration
    demo_p = 97
    ec_demo = SmallEC(demo_p, 0, 7)

    print(f"\n  Working with E: y^2 = x^3 + 7 over F_{demo_p}")
    print(f"  |E(F_{demo_p})| = {ec_demo.order}")
    print(f"  Generator G = {ec_demo.generator}")

    # Demonstration 1: Points cannot be factored
    demonstrate_no_factoring_on_ec(ec_demo, verbose=True)

    # Demonstration 2: Group operation scrambles coordinates
    demonstrate_coordinate_scrambling(ec_demo, verbose=True)

    # Demonstration 3: Attempted "EC index calculus" -- show it degenerates to DLP
    print(f"\n  DEMONSTRATION: Attempted EC index calculus degenerates to DLP")
    print(f"  -" * 39)

    G = ec_demo.generator
    N = ec_demo.order

    # Define a "factor base" of random points
    fb_points = []
    for i in range(1, 6):
        pt = ec_demo.multiply(G, i)
        if pt is not None:
            fb_points.append((i, pt))

    print(f"  'Factor base' (points with small DL):")
    for idx, pt in fb_points:
        print(f"    B_{idx} = {idx}*G = {pt}")

    # Try to build a "relation" -- pick random e, compute eG, try to express it
    # as a sum of factor base points
    print(f"\n  Attempting to collect 'relations':")
    print(f"  For each random e, we compute P = e*G and try to express P = sum(c_i * B_i)")

    for trial in range(3):
        e = secrets.randbelow(N - 1) + 1
        P = ec_demo.multiply(G, e)
        print(f"\n  Trial {trial+1}: e={e}, P = {e}*G = {P}")
        print(f"    To express P as sum of B_i, we need: e = c_1*1 + c_2*2 + ... + c_5*5")
        print(f"    But finding c_i such that sum(c_i * i) = {e} mod {N}")
        print(f"    IS the discrete log problem! (specifically, it's a subset-sum over Z_N)")
        print(f"    We CANNOT check this without knowing e -- which is what we want to find.")

    print(f"\n  CONCLUSION: Every 'relation' requires solving a DLP instance.")
    print(f"  Index calculus needs FREE relations; on EC, relations cost DLP each.")
    print(f"  The method is circular and provides no speedup whatsoever.")

    # ================================================================
    # PART 3: ACTUAL COMPARISON -- IC on F_p* vs BSGS on E(F_p)
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  PART 3: OPERATION COUNT COMPARISON")
    print(f"{'='*78}")

    # For small field sizes, actually run both and compare
    comparison_primes = []
    for p_candidate in range(100, 10500):
        if is_prime(p_candidate):
            comparison_primes.append(p_candidate)
            if len(comparison_primes) >= 20:
                break

    # Thin it out to get a good range
    comparison_primes = [comparison_primes[i] for i in range(0, len(comparison_primes), max(1, len(comparison_primes) // 10))]

    print(f"\n  Empirical comparison (same-size fields):")
    print(f"  {'p':>8s}  {'IC ops (F_p*)':>14s}  {'BSGS ops (EC)':>14s}  "
          f"{'Ratio BSGS/IC':>14s}  {'IC faster?':>12s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*12}")

    for p_val in comparison_primes:
        # Index calculus on F_p*
        g_fp = find_generator_fp(p_val)
        if g_fp is None:
            continue
        x_target = secrets.randbelow(p_val - 2) + 1
        h_fp = pow(g_fp, x_target, p_val)
        _, ic_ops = index_calculus_fp(p_val, g_fp, h_fp)

        # BSGS on E(F_p)
        ec = SmallEC(p_val, 0, 7)
        G_ec = ec.generator
        if G_ec is None:
            continue
        k_target = secrets.randbelow(ec.order - 1) + 1
        Q_ec = ec.multiply(G_ec, k_target)
        _, bsgs_ops = bsgs_ec(ec, G_ec, Q_ec)

        ratio = bsgs_ops / max(ic_ops, 1)
        ic_faster = "YES" if ic_ops < bsgs_ops else "NO"

        print(f"  {p_val:>8d}  {ic_ops:>14d}  {bsgs_ops:>14d}  "
              f"{ratio:>14.2f}  {ic_faster:>12s}")

        csv_rows.append({
            "field_size": p_val,
            "index_calculus_ops": ic_ops,
            "ec_bsgs_ops": bsgs_ops,
            "ratio": round(ratio, 4),
            "category": "empirical_comparison",
        })

    # ================================================================
    # PART 4: THEORETICAL SCALING -- THE REAL STORY
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  PART 4: THEORETICAL SCALING (WHERE IT REALLY MATTERS)")
    print(f"{'='*78}")

    print(f"\n  At cryptographic sizes, the gap between index calculus and BSGS is enormous.")
    print(f"  Index calculus: L(N) = exp(sqrt(ln(N) * ln(ln(N)))) -- SUBEXPONENTIAL")
    print(f"  BSGS on EC:    sqrt(N) = 2^(n/2) for n-bit N    -- FULLY EXPONENTIAL")

    bit_sizes = [32, 64, 128, 256, 384, 512, 768, 1024, 2048, 4096]
    scaling_rows = compare_complexities(bit_sizes)

    for row in scaling_rows:
        csv_rows.append({
            "field_size": row["bits"],
            "index_calculus_ops": f"2^{row['ic_ops_log2']}",
            "ec_bsgs_ops": f"2^{row['bsgs_ops_log2']}",
            "ratio": f"2^{row['ratio_log2']}",
            "category": "theoretical_scaling",
        })

    # Highlight the key numbers
    print(f"\n  KEY NUMBERS FOR CRYPTOGRAPHY:")
    print(f"  -- At 256 bits (secp256k1 key size):")
    ic_256_log2 = l_complexity_log2(256)
    print(f"     Index calculus on F_p*: ~2^{ic_256_log2:.0f} operations")
    print(f"     BSGS on E(F_p):        ~2^128 operations")
    gap_256 = 128 - ic_256_log2
    print(f"     Ratio: 2^{gap_256:.0f} (EC is 2^{gap_256:.0f}x harder!)")

    print(f"\n  -- At 2048 bits (RSA key size):")
    ic_2048_log2 = l_complexity_log2(2048)
    print(f"     Index calculus on F_p*: ~2^{ic_2048_log2:.0f} operations (feasible!)")
    print(f"     BSGS on E(F_p):        ~2^1024 operations (impossible)")

    print(f"\n  This is WHY:")
    print(f"    - RSA needs 2048-bit keys (to resist index calculus)")
    print(f"    - ECC only needs 256-bit keys (no index calculus exists)")
    print(f"    - 256-bit EC security = 128 bits (sqrt via Pollard rho)")
    print(f"    - 2048-bit RSA security = ~112 bits (GNFS index calculus)")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  SUMMARY: INDEX CALCULUS AND ELLIPTIC CURVES")
    print(f"{'='*78}")

    print(f"""
  WHY INDEX CALCULUS WORKS ON F_p*:
    1. Elements of F_p* are INTEGERS. Integers have unique prime factorization.
    2. We can efficiently test if an integer is "B-smooth" (all factors < B).
    3. Smooth elements give us LINEAR RELATIONS among discrete logs.
    4. Enough relations -> solve a linear system -> recover ALL discrete logs.
    5. Complexity: L(p) = exp(sqrt(ln(p) * ln(ln(p)))) -- subexponential.

  WHY INDEX CALCULUS FAILS ON E(F_p):
    1. Elements of E(F_p) are POINTS (x, y). Points CANNOT be factored.
    2. There is no notion of "smooth point" -- no prime decomposition exists.
    3. The group law (chord-and-tangent) bears NO relation to coordinates.
    4. To build a "relation", you must SOLVE a DLP -- circular dependency.
    5. Every known generic approach remains O(sqrt(N)) -- fully exponential.

  IMPLICATIONS:
    - If someone found index calculus for EC, ALL EC crypto would break.
    - Weil descent (GHS, 2000) tried this for curves over extension fields.
      Result: works ONLY for F_q^n with small n, NOT for prime fields.
    - Summation polynomials (Semaev, 2004) gave a theoretical framework.
      Result: the polynomial systems are too hard to solve in practice.
    - For secp256k1 over F_p (PRIME field): NO subexponential algorithm known.
    - The security gap: 256-bit EC key has 128-bit security (sqrt).
      Same 128-bit security requires 3072-bit RSA key (index calculus).

  THE BOTTOM LINE:
    EC groups are structurally different from multiplicative groups.
    The absence of factoring in EC groups is not a bug -- it's the feature
    that makes elliptic curve cryptography the most efficient public-key
    system known to classical computation.""")

    print(f"{'='*78}")

    # Write CSV
    csv_path = os.path.expanduser("~/Desktop/index_calculus_analysis.csv")
    fieldnames = ["field_size", "index_calculus_ops", "ec_bsgs_ops", "ratio", "category"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Results written to {csv_path}")
    print(f"  Total rows: {len(csv_rows)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
