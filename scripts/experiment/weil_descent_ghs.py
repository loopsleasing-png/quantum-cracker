"""Weil Descent / GHS (Gaudry-Hess-Smart) Attack Analysis.

The most sophisticated algebraic attack ever attempted against ECDLP.

Weil Descent Theory:
  Given an elliptic curve E defined over an extension field F_{q^n},
  Weil descent (also called Weil restriction of scalars) maps E/F_{q^n}
  to a higher-dimensional abelian variety A/F_q. The key idea:

    E/F_{q^n}  --->  A/F_q  (dimension n variety)

  If A is (isogenous to) the Jacobian of a curve C/F_q of genus g,
  then the DLP in E(F_{q^n}) transfers to the DLP in Jac(C)(F_q).

  For hyperelliptic Jacobians of genus g, index calculus methods
  (Adleman-DeMarrais-Huang, Enge-Gaudry, Gaudry) solve the DLP in
  subexponential time when g is small relative to log(q).

GHS Attack (Gaudry-Hess-Smart, 2002):
  For E/F_{2^n}, the Weil descent produces a hyperelliptic curve C/F_2
  of genus g. In the worst case:

    g = 2^(n/2 - 1)    (for odd-degree extensions n)

  When n is composite (n = n1 * n2), intermediate field descent via
  F_{2^n} -> F_{2^{n1}} can yield much smaller genus, making the
  attack feasible for specific curves.

  For n prime (e.g., 163, 233, 283): the genus is astronomical,
  and the attack is worse than Pollard rho (O(sqrt(N))).

Prime Field Immunity:
  secp256k1 is defined over F_p where p is prime -- NOT an extension
  field. The Weil restriction of F_p to F_p is the identity map.
  There is no "descent" possible: the curve stays exactly the same.
  This is a fundamental structural protection that makes GHS
  completely inapplicable to prime-field curves.

History:
  - 2002: GHS published; showed certain NIST binary curves over F_{2^n}
    with composite n could be attacked
  - 2004: Menezes-Qu showed most random binary curves resist GHS
  - 2009: NIST began recommending prime curves over binary curves
  - 2014: NSA Suite B dropped binary curves entirely
  - 2023: NIST deprecated all binary curves in SP 800-186

This script demonstrates GHS genus analysis on small binary field
curves, proves prime field immunity, and compares all standard curves.
"""

import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")


# ================================================================
# BINARY FIELD ARITHMETIC: F_{2^n} represented as polynomials over GF(2)
# Elements are integers where bit i represents the coefficient of x^i.
# ================================================================

class BinaryField:
    """Arithmetic in F_{2^n} using polynomial representation.

    Elements are integers; bit i = coefficient of x^i in GF(2)[x].
    Reduction is modulo an irreducible polynomial of degree n.
    """

    # Irreducible polynomials for small extension degrees.
    # Stored as integer bitmask: e.g., x^8 + x^4 + x^3 + x + 1 = 0x11B
    IRREDUCIBLES = {
        2:   0b111,                 # x^2 + x + 1
        3:   0b1011,                # x^3 + x + 1
        4:   0b10011,               # x^4 + x + 1
        5:   0b100101,              # x^5 + x^2 + 1
        6:   0b1000011,             # x^6 + x + 1
        7:   0b10000011,            # x^7 + x + 1
        8:   0b100011011,           # x^8 + x^4 + x^3 + x + 1
        9:   0b1000010001,          # x^9 + x^4 + 1
        10:  0b10000001001,         # x^10 + x^3 + 1
        11:  0b100000000101,        # x^11 + x^2 + 1
        12:  0b1000001010011,       # x^12 + x^6 + x^4 + x + 1
        13:  0b10000000011011,      # x^13 + x^4 + x^3 + x + 1
        14:  0b100010000000011,     # x^14 + x^10 + x + 1 -- actually x^14+x^5+1
        15:  0b1000000000000011,    # x^15 + x + 1
        16:  0b10000000000101101,   # x^16 + x^5 + x^3 + x^2 + 1
    }

    def __init__(self, n):
        if n not in self.IRREDUCIBLES:
            raise ValueError(f"No irreducible polynomial stored for n={n}")
        self.n = n
        self.modulus = self.IRREDUCIBLES[n]
        self.order = (1 << n) - 1  # |F_{2^n}*| = 2^n - 1

    def add(self, a, b):
        """Addition in F_{2^n} = XOR."""
        return a ^ b

    def sub(self, a, b):
        """Subtraction = addition in characteristic 2."""
        return a ^ b

    def neg(self, a):
        """Negation = identity in characteristic 2."""
        return a

    def mul(self, a, b):
        """Multiplication in F_{2^n} via shift-and-XOR with reduction."""
        result = 0
        while b:
            if b & 1:
                result ^= a
            a <<= 1
            if a & (1 << self.n):
                a ^= self.modulus
            b >>= 1
        return result

    def sqr(self, a):
        """Squaring (same as mul but slightly faster)."""
        return self.mul(a, a)

    def inv(self, a):
        """Multiplicative inverse via Fermat: a^{-1} = a^{2^n - 2}."""
        if a == 0:
            raise ZeroDivisionError("inverse of zero in F_{2^n}")
        # a^{-1} = a^{2^n - 2} by Fermat's little theorem
        exp = (1 << self.n) - 2
        result = 1
        base = a
        while exp:
            if exp & 1:
                result = self.mul(result, base)
            base = self.sqr(base)
            exp >>= 1
        return result

    def div(self, a, b):
        """Division: a / b = a * b^{-1}."""
        return self.mul(a, self.inv(b))

    def pow(self, a, exp):
        """Exponentiation by squaring."""
        if exp == 0:
            return 1
        result = 1
        base = a
        while exp:
            if exp & 1:
                result = self.mul(result, base)
            base = self.sqr(base)
            exp >>= 1
        return result

    def sqrt(self, a):
        """Square root in F_{2^n}: sqrt(a) = a^{2^{n-1}}."""
        # In characteristic 2, Frobenius is x -> x^2, so x^{2^n} = x.
        # Therefore sqrt(a) = a^{2^{n-1}}.
        result = a
        for _ in range(self.n - 1):
            result = self.sqr(result)
        return result

    def trace(self, a):
        """Absolute trace: Tr(a) = a + a^2 + a^{2^2} + ... + a^{2^{n-1}}.

        Returns 0 or 1 (element of F_2).
        """
        t = a
        x = a
        for _ in range(self.n - 1):
            x = self.sqr(x)
            t ^= x
        return t & 1

    def random_element(self, rng):
        """Random nonzero element."""
        while True:
            val = rng.integers(1, 1 << self.n)
            if val != 0:
                return int(val)

    def enumerate_elements(self):
        """Yield all elements of F_{2^n} (including 0)."""
        for i in range(1 << self.n):
            yield i


# ================================================================
# ELLIPTIC CURVE OVER F_{2^n}
# Using short Weierstrass form for char 2:
#   y^2 + xy = x^3 + ax^2 + b  (non-supersingular)
# This is the standard form for binary curves (NIST B-xxx, K-xxx).
# ================================================================

class BinaryEC:
    """Elliptic curve over F_{2^n} in the form y^2 + xy = x^3 + ax^2 + b."""

    def __init__(self, field, a, b):
        self.F = field
        self.a = a
        self.b = b
        self._order = None
        self._points = None

    def is_on_curve(self, P):
        """Check if point P = (x, y) is on the curve."""
        if P is None:
            return True
        x, y = P
        F = self.F
        # y^2 + xy = x^3 + ax^2 + b
        lhs = F.add(F.sqr(y), F.mul(x, y))
        rhs = F.add(F.add(F.mul(F.sqr(x), x), F.mul(self.a, F.sqr(x))), self.b)
        return lhs == rhs

    def add(self, P, Q):
        """Point addition on y^2 + xy = x^3 + ax^2 + b."""
        if P is None:
            return Q
        if Q is None:
            return P
        F = self.F
        x1, y1 = P
        x2, y2 = Q

        if x1 == x2:
            if F.add(y1, y2) == x1:
                # P + (-P) = O  (since -P = (x, x+y) in this form)
                return None
            if y1 == y2:
                # Doubling
                if x1 == 0:
                    return None
                # lambda = x1 + y1/x1
                lam = F.add(x1, F.div(y1, x1))
                x3 = F.add(F.add(F.sqr(lam), lam), self.a)
                y3 = F.add(F.mul(lam, F.add(x1, x3)), F.add(x3, y1))
                # Actually: x3 = lam^2 + lam + a
                # y3 = x1^2 + (lam+1)*x3
                x3 = F.add(F.add(F.sqr(lam), lam), self.a)
                y3 = F.add(F.sqr(x1), F.mul(F.add(lam, 1), x3))
                return (x3, y3)

        # General addition: P != Q, x1 != x2
        lam = F.div(F.add(y1, y2), F.add(x1, x2))
        x3 = F.add(F.add(F.add(F.add(F.sqr(lam), lam), x1), x2), self.a)
        y3 = F.add(F.add(F.mul(lam, F.add(x1, x3)), x3), y1)
        return (x3, y3)

    def neg(self, P):
        """Negation: -(x, y) = (x, x + y) for y^2 + xy = x^3 + ax^2 + b."""
        if P is None:
            return None
        x, y = P
        return (x, self.F.add(x, y))

    def multiply(self, P, k):
        """Scalar multiplication via double-and-add."""
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

    def enumerate_points(self):
        """Brute-force enumerate all points (only for small fields)."""
        if self._points is not None:
            return self._points
        F = self.F
        pts = [None]  # point at infinity
        for x in F.enumerate_elements():
            for y in F.enumerate_elements():
                if self.is_on_curve((x, y)):
                    pts.append((x, y))
        self._points = pts
        self._order = len(pts)
        return pts

    @property
    def order(self):
        if self._order is None:
            self.enumerate_points()
        return self._order

    def find_generator(self):
        """Find a generator of the curve group (brute force for small curves)."""
        pts = self.enumerate_points()
        N = self.order
        for pt in pts[1:]:
            if self.multiply(pt, N) is not None:
                continue
            is_gen = True
            # Check that pt does not have smaller order
            temp = N
            d = 2
            while d * d <= temp:
                if temp % d == 0:
                    if self.multiply(pt, N // d) is None:
                        is_gen = False
                        break
                    while temp % d == 0:
                        temp //= d
                d += 1
            if temp > 1 and is_gen:
                if self.multiply(pt, N // temp) is None:
                    is_gen = False
            if is_gen:
                return pt
        return pts[1] if len(pts) > 1 else None


# ================================================================
# GHS GENUS COMPUTATION
# ================================================================

def ghs_genus_bound(n, q=2):
    """Compute the genus bound for GHS Weil descent.

    For E/F_{q^n} with the GHS attack:
    - If n is odd: genus g <= 2^{(n-1)/2}
    - If n is even: genus g <= 2^{n/2 - 1}  (slightly better)
    - For composite n = n1*n2: can do partial descent F_{q^n} -> F_{q^{n1}},
      getting genus ~ 2^{(n2-1)/2} which may be much smaller.

    Returns (worst_case_genus, best_composite_genus, factorizations).
    """
    if n <= 1:
        return 1, 1, []

    # Worst case genus (full descent to base field)
    if n % 2 == 0:
        worst_genus = 1 << (n // 2 - 1)
    else:
        worst_genus = 1 << ((n - 1) // 2)

    # For composite n, try all factorizations
    factorizations = []
    best_genus = worst_genus

    for d in range(2, n):
        if n % d == 0:
            n1 = d
            n2 = n // d
            # Partial descent: E/F_{2^n} -> C/F_{2^{n1}}
            # Genus of resulting curve over F_{2^{n1}}
            if n2 % 2 == 0:
                partial_genus = 1 << (n2 // 2 - 1)
            else:
                partial_genus = 1 << ((n2 - 1) // 2)

            factorizations.append((n1, n2, partial_genus))
            if partial_genus < best_genus:
                best_genus = partial_genus

    return worst_genus, best_genus, factorizations


def index_calculus_complexity(g, q_base, n_base=1):
    """Estimate index calculus complexity on genus-g hyperelliptic Jacobian over F_{q_base^{n_base}}.

    For genus g Jacobian over F_q:
      L_{q^g}[1/2, c] = exp(c * (log(q^g))^{1/2} * (log log(q^g))^{1/2})

    This is subexponential when g is small relative to log(q).
    For large g, it degrades to worse than sqrt(group order).

    Returns log2 of the estimated operation count.
    """
    q = q_base ** n_base
    # Group order ~ q^g
    log_group = g * math.log(q) if q > 1 else 1
    loglog_group = math.log(max(log_group, 2))

    # L[1/2] complexity
    c = 1.0  # constant factor (varies by algorithm)
    L_half = c * math.sqrt(log_group * loglog_group)

    return L_half / math.log(2)  # convert to log2


def pollard_rho_complexity(group_order_bits):
    """Pollard rho: O(sqrt(N)) = O(2^{n/2}) where N ~ 2^n."""
    return group_order_bits / 2.0


# ================================================================
# DEMONSTRATIONS
# ================================================================

def demonstrate_binary_field_curve(n):
    """Construct and analyze an EC over F_{2^n}."""
    F = BinaryField(n)
    q = 1 << n

    print(f"\n  F_{{2^{n}}} = GF({q})")
    print(f"    Field elements: {q}")
    print(f"    Multiplicative order: {q - 1}")

    # Find a valid curve y^2 + xy = x^3 + ax^2 + b over F_{2^n}
    # Need b != 0 for non-singular curve
    rng = np.random.default_rng(42 + n)
    curve = None
    a_val = 0
    b_val = 0

    for attempt in range(100):
        a_val = int(rng.integers(0, q))
        b_val = int(rng.integers(1, q))  # b != 0
        ec = BinaryEC(F, a_val, b_val)

        # For small fields, enumerate and check order
        if n <= 12:
            pts = ec.enumerate_points()
            N = ec.order
            # Valid curve: check Hasse bound |N - (q+1)| <= 2*sqrt(q)
            if abs(N - (q + 1)) <= 2 * math.isqrt(q) + 1 and N > 4:
                curve = ec
                break
        else:
            # For larger fields, just trust the construction
            curve = ec
            break

    if curve is None:
        print(f"    [Could not find valid curve for n={n}]")
        return None

    N = curve.order if n <= 12 else None

    print(f"    Curve: y^2 + xy = x^3 + {a_val:#x}*x^2 + {b_val:#x}")
    if N is not None:
        print(f"    |E(F_{{2^{n}}})| = {N}")
        hasse_deviation = N - (q + 1)
        print(f"    Hasse: N - (q+1) = {hasse_deviation}, bound = +/- {2*math.isqrt(q)}")

    # GHS genus analysis
    worst_genus, best_genus, factorizations = ghs_genus_bound(n)
    print(f"    GHS worst-case genus (full descent to F_2): {worst_genus}")

    if factorizations:
        print(f"    Composite n={n} factorizations:")
        for n1, n2, pg in factorizations:
            print(f"      {n} = {n1} x {n2}: partial descent genus = {pg}")
        print(f"    Best composite genus: {best_genus}")
    else:
        print(f"    n={n} is prime -- no partial descent available")

    # Complexity comparison
    group_bits = math.log2(q) if N is None else math.log2(max(N, 2))
    rho_bits = pollard_rho_complexity(group_bits)
    ic_bits_worst = index_calculus_complexity(worst_genus, 2)
    ic_bits_best = index_calculus_complexity(best_genus, 2) if best_genus < worst_genus else ic_bits_worst

    print(f"    Pollard rho: ~2^{rho_bits:.1f} ops")
    print(f"    Index calculus (worst genus): ~2^{ic_bits_worst:.1f} ops")
    if best_genus < worst_genus:
        print(f"    Index calculus (best genus):  ~2^{ic_bits_best:.1f} ops")

    feasible = ic_bits_best < rho_bits
    print(f"    GHS advantage over rho? {'YES -- attack is faster' if feasible else 'NO -- rho is still better'}")

    return {
        "n": n,
        "q": q,
        "order": N,
        "worst_genus": worst_genus,
        "best_genus": best_genus,
        "is_composite": len(factorizations) > 0,
        "rho_bits": rho_bits,
        "ic_bits_worst": ic_bits_worst,
        "ic_bits_best": ic_bits_best,
        "ghs_feasible": feasible,
    }


def demonstrate_small_field_dlp(n):
    """Actually solve DLP on a small binary curve to show it works."""
    if n > 12:
        return None

    F = BinaryField(n)
    q = 1 << n
    rng = np.random.default_rng(137 + n)

    # Find a curve with a generator
    for attempt in range(50):
        a_val = int(rng.integers(0, q))
        b_val = int(rng.integers(1, q))
        ec = BinaryEC(F, a_val, b_val)
        pts = ec.enumerate_points()
        N = ec.order
        if N < 8:
            continue
        G = ec.find_generator()
        if G is not None:
            break
    else:
        return None

    # Pick a random target
    k_target = int(rng.integers(1, N))
    Q = ec.multiply(G, k_target)

    # Solve by brute force (simulating what index calculus would do)
    t0 = time.time()
    k_found = None
    for k in range(1, N + 1):
        if ec.multiply(G, k) == Q:
            k_found = k
            break
    dt = (time.time() - t0) * 1000

    success = k_found == k_target or (k_found is not None and ec.multiply(G, k_found) == Q)
    return {
        "n": n,
        "order": N,
        "target": k_target,
        "found": k_found,
        "success": success,
        "time_ms": dt,
    }


# ================================================================
# STANDARD CURVE COMPARISON TABLE
# ================================================================

STANDARD_CURVES = [
    # (name, field_type, base_field, extension_degree, group_order_bits, status)
    # Prime field curves
    ("secp256k1",    "prime",  "F_p",     1,  256, "Active (Bitcoin)"),
    ("P-256",        "prime",  "F_p",     1,  256, "Active (NIST)"),
    ("P-384",        "prime",  "F_p",     1,  384, "Active (NIST)"),
    ("P-521",        "prime",  "F_p",     1,  521, "Active (NIST)"),
    ("Ed25519",      "prime",  "F_p",     1,  253, "Active (modern)"),
    ("Ed448",        "prime",  "F_p",     1,  446, "Active (modern)"),
    ("Curve25519",   "prime",  "F_p",     1,  253, "Active (TLS)"),
    ("brainpoolP256r1","prime","F_p",     1,  256, "Active (EU)"),
    ("brainpoolP384r1","prime","F_p",     1,  384, "Active (EU)"),
    ("brainpoolP512r1","prime","F_p",     1,  512, "Active (EU)"),
    # Binary field curves (many now deprecated)
    ("B-163",        "binary", "F_2",   163,  163, "Deprecated (NIST 2023)"),
    ("B-233",        "binary", "F_2",   233,  233, "Deprecated (NIST 2023)"),
    ("B-283",        "binary", "F_2",   283,  283, "Deprecated (NIST 2023)"),
    ("B-409",        "binary", "F_2",   409,  409, "Deprecated (NIST 2023)"),
    ("B-571",        "binary", "F_2",   571,  571, "Deprecated (NIST 2023)"),
    ("K-163",        "binary", "F_2",   163,  163, "Deprecated (NIST 2023)"),
    ("K-233",        "binary", "F_2",   233,  233, "Deprecated (NIST 2023)"),
    ("K-283",        "binary", "F_2",   283,  283, "Deprecated (NIST 2023)"),
    ("K-409",        "binary", "F_2",   409,  409, "Deprecated (NIST 2023)"),
    ("K-571",        "binary", "F_2",   571,  571, "Deprecated (NIST 2023)"),
    # Historical binary curves with composite extension degree (worst case)
    ("sect131r1",    "binary", "F_2",   131,  131, "Removed (composite n)"),
    ("sect163k1",    "binary", "F_2",   163,  163, "Deprecated"),
    ("sect239k1",    "binary", "F_2",   239,  239, "Removed (composite n)"),
    ("sect113r1",    "binary", "F_2",   113,  113, "Removed (composite n)"),
]


def analyze_standard_curves():
    """Analyze GHS vulnerability for all standard curves."""
    results = []

    for name, ftype, base, ext_deg, bits, status in STANDARD_CURVES:
        if ftype == "prime":
            # Prime field: Weil descent is trivial (identity)
            worst_genus = 1
            best_genus = 1
            ghs_bits = float("inf")
            rho_bits = bits / 2.0
            vulnerable = "IMMUNE"
            reason = "Prime field (no extension structure)"
        else:
            # Binary field: compute GHS genus
            worst_genus, best_genus, factorizations = ghs_genus_bound(ext_deg)
            rho_bits = bits / 2.0
            ghs_bits = index_calculus_complexity(best_genus, 2)
            vulnerable = "YES" if ghs_bits < rho_bits else "NO"
            if ext_deg in (131, 239, 113):
                # These have composite extension degree
                reason = f"Composite n={ext_deg}, genus reducible"
            elif all(ext_deg % d != 0 for d in range(2, ext_deg)):
                reason = f"Prime n={ext_deg}, genus too large"
            else:
                reason = f"n={ext_deg}, genus analysis required"

        results.append({
            "curve": name,
            "field_type": ftype,
            "extension_degree": ext_deg,
            "group_bits": bits,
            "worst_genus": worst_genus,
            "best_genus": best_genus,
            "rho_log2": f"{rho_bits:.1f}",
            "ghs_log2": f"{ghs_bits:.1f}" if ghs_bits < 1e10 else "N/A",
            "ghs_vulnerable": vulnerable,
            "reason": reason,
            "status": status,
        })

    return results


def is_prime(n):
    """Simple primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


# ================================================================
# MAIN
# ================================================================

def main():
    print()
    print("=" * 78)
    print("  WEIL DESCENT / GHS ATTACK ANALYSIS")
    print("  The most sophisticated algebraic attack on ECDLP")
    print("=" * 78)

    csv_rows = []

    # ================================================================
    # PART 1: Binary field curve construction and analysis
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  PART 1: BINARY FIELD CURVES -- F_{{2^n}} ANALYSIS")
    print(f"{'='*78}")

    small_results = []
    for n in [4, 6, 8, 10, 12, 16]:
        result = demonstrate_binary_field_curve(n)
        if result:
            small_results.append(result)

    # ================================================================
    # PART 2: DLP on small binary curves (proof it works)
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  PART 2: DLP SOLUTIONS ON SMALL BINARY CURVES")
    print(f"{'='*78}")

    print(f"\n  {'n':>4s}  {'|E|':>8s}  {'Target':>8s}  {'Found':>8s}  {'OK?':>5s}  {'Time':>8s}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*8}")

    for n in [4, 6, 8, 10, 12]:
        dlp_result = demonstrate_small_field_dlp(n)
        if dlp_result:
            print(f"  {dlp_result['n']:>4d}  {dlp_result['order']:>8d}  "
                  f"{dlp_result['target']:>8d}  {dlp_result['found']:>8}  "
                  f"{'yes' if dlp_result['success'] else 'no':>5s}  "
                  f"{dlp_result['time_ms']:>6.1f}ms")

    # ================================================================
    # PART 3: GHS genus analysis for NIST binary extension degrees
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  PART 3: GHS GENUS ANALYSIS FOR NIST EXTENSION DEGREES")
    print(f"{'='*78}")

    nist_degrees = [163, 233, 283, 409, 571]
    # Also analyze some composite degrees that were historically used
    composite_degrees = [113, 131, 176, 239, 272]

    print(f"\n  NIST binary curve extension degrees (all prime):")
    print(f"  {'n':>5s}  {'Prime?':>7s}  {'Worst Genus':>14s}  {'Best Genus':>14s}  "
          f"{'Rho bits':>10s}  {'GHS bits':>10s}  {'GHS useful?':>12s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*12}")

    for n in nist_degrees:
        wg, bg, facts = ghs_genus_bound(n)
        rho = n / 2.0
        ghs = index_calculus_complexity(bg, 2)
        p = is_prime(n)
        useful = "YES" if ghs < rho else "NO"
        # For display: use scientific notation for huge genus
        wg_str = f"2^{int(math.log2(wg))}" if wg > 1000 else str(wg)
        bg_str = f"2^{int(math.log2(bg))}" if bg > 1000 else str(bg)
        print(f"  {n:>5d}  {'yes':>7s if p else 'no':>7s}  {wg_str:>14s}  {bg_str:>14s}  "
              f"{rho:>10.1f}  {ghs:>10.1f}  {useful:>12s}")

    print(f"\n  Historically used composite extension degrees:")
    print(f"  {'n':>5s}  {'Prime?':>7s}  {'Factors':>18s}  {'Best Genus':>14s}  "
          f"{'Rho bits':>10s}  {'GHS bits':>10s}  {'GHS useful?':>12s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*18}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*12}")

    for n in composite_degrees:
        wg, bg, facts = ghs_genus_bound(n)
        rho = n / 2.0
        ghs = index_calculus_complexity(bg, 2)
        p = is_prime(n)
        useful = "YES" if ghs < rho else "NO"

        # Show factorizations
        factor_strs = []
        for n1, n2, pg in facts[:3]:
            factor_strs.append(f"{n1}x{n2}")
        factor_str = ", ".join(factor_strs) if factor_strs else "prime"

        bg_str = f"2^{int(math.log2(bg))}" if bg > 1000 else str(bg)
        print(f"  {n:>5d}  {'yes' if p else 'no':>7s}  {factor_str:>18s}  {bg_str:>14s}  "
              f"{rho:>10.1f}  {ghs:>10.1f}  {useful:>12s}")

    # ================================================================
    # PART 4: PRIME FIELD IMMUNITY PROOF
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  PART 4: PRIME FIELD IMMUNITY -- WHY secp256k1 IS SAFE")
    print(f"{'='*78}")

    secp256k1_p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    secp256k1_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    print(f"""
  secp256k1 is defined over F_p where p is a 256-bit prime.

  Key structural facts:
    Field:       F_p (a prime field, NOT an extension field)
    p =          {secp256k1_p}
    p is prime:  YES (2^256 - 2^32 - 977)
    Extension:   F_p has extension degree 1 over itself
    Group order: {secp256k1_n}

  Weil Descent Analysis:
    The Weil restriction of scalars Res_{{F_p/F_p}}(E) is just E itself.
    There is no non-trivial "descent" -- the base field IS the field.

    - No extension structure to exploit
    - No factorization of extension degree (degree = 1)
    - The resulting variety has genus 1 (the original curve)
    - Index calculus on genus-1 Jacobian = original ECDLP

  Formal proof of immunity:
    1. E is defined over F_p (prime, not F_{{p^n}} for any n > 1)
    2. Weil restriction Res_{{F_p/F_p}} is the identity functor
    3. The descended variety is E itself, genus 1
    4. No genus amplification occurs
    5. Index calculus methods require genus >> 1 to gain advantage
    6. Therefore GHS attack provides ZERO speedup

  The same applies to ALL prime-field curves:
    P-256, P-384, P-521, Ed25519, Ed448, Curve25519, brainpool, etc.

  This is why the cryptographic community moved from binary to prime curves.
""")

    # ================================================================
    # PART 5: COMPREHENSIVE COMPARISON TABLE
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  PART 5: COMPREHENSIVE STANDARD CURVE GHS ANALYSIS")
    print(f"{'='*78}")

    curve_results = analyze_standard_curves()

    print(f"\n  {'Curve':18s}  {'Type':>6s}  {'Ext.n':>5s}  {'Bits':>4s}  "
          f"{'Best Genus':>12s}  {'Rho':>7s}  {'GHS':>7s}  {'Vuln?':>8s}  {'Status'}")
    print(f"  {'-'*18}  {'-'*6}  {'-'*5}  {'-'*4}  "
          f"{'-'*12}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*20}")

    for r in curve_results:
        bg = r["best_genus"]
        if bg > 1000:
            bg_str = f"2^{int(math.log2(bg))}"
        else:
            bg_str = str(bg)

        print(f"  {r['curve']:18s}  {r['field_type']:>6s}  {r['extension_degree']:>5d}  "
              f"{r['group_bits']:>4d}  {bg_str:>12s}  {r['rho_log2']:>7s}  "
              f"{r['ghs_log2']:>7s}  {r['ghs_vulnerable']:>8s}  {r['status']}")

    # CSV rows
    for r in curve_results:
        csv_rows.append({
            "curve": r["curve"],
            "field_type": r["field_type"],
            "extension_degree": r["extension_degree"],
            "group_bits": r["group_bits"],
            "worst_genus": r["worst_genus"],
            "best_genus": r["best_genus"],
            "pollard_rho_log2": r["rho_log2"],
            "ghs_index_calculus_log2": r["ghs_log2"],
            "ghs_vulnerable": r["ghs_vulnerable"],
            "reason": r["reason"],
            "status": r["status"],
        })

    # ================================================================
    # PART 6: HISTORICAL CONTEXT
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  PART 6: HISTORICAL IMPACT OF GHS")
    print(f"{'='*78}")

    print(f"""
  Timeline of the GHS attack and its consequences:

  2000: Gaudry-Hess-Smart propose Weil descent attack on binary curves
        - Showed E/F_{{2^n}} can be transferred to hyperelliptic Jacobians
        - Initial analysis: some NIST curves could be vulnerable

  2002: Full GHS paper published
        - For composite n: partial descent yields manageable genus
        - For prime n (163, 233, 283, 409, 571): genus too large
        - NIST curves were deliberately chosen with prime extension degrees

  2004: Menezes-Qu analysis
        - For random curves over F_{{2^n}} with prime n, GHS fails
        - Only specially constructed "weak" curves are vulnerable
        - NIST binary curves are safe from GHS specifically

  2009: Growing unease about binary curves
        - Side-channel attacks easier on binary field arithmetic
        - GHS showed binary curves have "extra attack surface"
        - NIST begins recommending prime curves for new deployments

  2014: NSA Suite B transition
        - Drops ALL binary curves
        - Only P-256 and P-384 remain
        - Signals death knell for binary curve cryptography

  2023: NIST SP 800-186
        - Formally deprecates all 10 binary curves (B-xxx and K-xxx)
        - Only prime curves (P-256, P-384, P-521) and Edwards curves remain
        - Binary curves may not be used for new federal systems

  Key lesson:
    GHS was the closest anyone came to a subexponential classical attack
    on ECDLP. It ultimately failed against properly chosen curves, but
    it demonstrated that binary extension field curves carry inherent
    structural risk that prime field curves do not.

    secp256k1 was NEVER at risk:
    - Defined over a prime field F_p (extension degree 1)
    - Weil descent is the identity -- no attack surface
    - Embedding degree is huge -- MOV/Frey-Ruck also inapplicable
    - The ONLY known efficient attack is Shor's algorithm (quantum)
""")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")

    n_prime_curves = sum(1 for r in curve_results if r["field_type"] == "prime")
    n_binary_curves = sum(1 for r in curve_results if r["field_type"] == "binary")
    n_immune = sum(1 for r in curve_results if r["ghs_vulnerable"] == "IMMUNE")
    n_vulnerable = sum(1 for r in curve_results if r["ghs_vulnerable"] == "YES")
    n_safe_binary = sum(1 for r in curve_results
                        if r["field_type"] == "binary" and r["ghs_vulnerable"] == "NO")

    print(f"""
  Curves analyzed: {len(curve_results)}
    Prime field (immune by construction): {n_immune}
    Binary field (safe due to prime ext. degree): {n_safe_binary}
    Binary field (potentially vulnerable): {n_vulnerable}

  GHS Attack Viability:
    - Prime field curves: COMPLETELY IMMUNE (no extension to descend)
    - Binary curves, prime n: genus 2^{{(n-1)/2}} >> sqrt(N), attack WORSE than rho
    - Binary curves, composite n: genus may be manageable, attack POSSIBLE

  secp256k1 Verdict:
    Field type:        Prime (F_p)
    Extension degree:  1 (trivial)
    GHS genus:         1 (no amplification)
    GHS complexity:    Same as ECDLP (no speedup)
    Vulnerability:     NONE

  The GHS attack was a landmark in cryptanalysis -- it showed that the
  algebraic structure of the base field matters fundamentally. But for
  prime-field curves like secp256k1, the attack surface simply does not
  exist. The curve lives in a prime field with no extension structure
  to exploit, and no Weil descent can change that.
""")
    print("=" * 78)

    # Write CSV
    csv_path = os.path.expanduser("~/Desktop/weil_descent_analysis.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n  Results written to {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
