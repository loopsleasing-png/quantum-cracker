"""MOV (Menezes-Okamoto-Vanstone) Pairing Attack.

The MOV attack reduces the EC discrete log problem (ECDLP) to the
finite field discrete log problem (FFDLP) via the Weil pairing.

Key idea: if the embedding degree k of the curve is small, we can
map the EC group into F_{p^k}* where DLP is easier (via index calculus).

For secp256k1: the embedding degree is HUGE (essentially p-1), making
MOV attack infeasible. But for poorly chosen curves, k can be small.

This script:
1. Finds the embedding degree of small EC curves
2. Attempts MOV attack when k is small
3. Shows why secp256k1 is immune (embedding degree analysis)

This is a REAL attack that has broken weak elliptic curves in practice.
"""

import math
import secrets
import sys
import time

sys.path.insert(0, "src")


class SmallEC:
    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b
        self._order = None
        self._gen = None
        self._points = None

    @property
    def order(self):
        if self._order is None:
            self._enumerate()
        return self._order

    @property
    def generator(self):
        if self._gen is None:
            self._find_gen()
        return self._gen

    def _enumerate(self):
        pts = [None]
        p = self.p
        qr = {}
        for y in range(p):
            qr.setdefault((y * y) % p, []).append(y)
        for x in range(p):
            rhs = (x * x * x + self.a * x + self.b) % p
            if rhs in qr:
                for y in qr[rhs]:
                    pts.append((x, y))
        self._points = pts
        self._order = len(pts)

    def _find_gen(self):
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
            if k & 1: result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result


def find_embedding_degree(p, n, max_k=1000):
    """Find smallest k such that n | (p^k - 1).

    The embedding degree k is the smallest positive integer where
    the group of n-th roots of unity embeds into F_{p^k}*.
    """
    if n <= 1:
        return 1

    pk_mod_n = 1
    for k in range(1, max_k + 1):
        pk_mod_n = (pk_mod_n * p) % n
        if pk_mod_n == 1:
            return k
    return None  # embedding degree > max_k


def baby_step_giant_step_ff(g, h, p, n):
    """BSGS for discrete log in F_p*: find x such that g^x = h mod p."""
    m = int(math.isqrt(n)) + 1

    # Baby steps: g^j for j in [0, m)
    baby = {}
    gj = 1
    for j in range(m):
        baby[gj] = j
        gj = (gj * g) % p

    # Giant step factor: g^(-m)
    gm_inv = pow(g, p - 1 - m, p)

    # Giant steps
    gamma = h
    for i in range(m):
        if gamma in baby:
            x = (baby[gamma] + i * m) % n
            # Verify
            if pow(g, x, p) == h % p:
                return x
        gamma = (gamma * gm_inv) % p

    return None


def simplified_weil_pairing(ec, P, Q, n):
    """Simplified Weil pairing computation.

    For educational purposes. Returns e(P, Q) in F_p.
    Only works when P and Q are linearly independent n-torsion points.

    The real Weil pairing uses Miller's algorithm on divisors.
    We approximate by using the Tate pairing for small curves.
    """
    p = ec.p

    # For supersingular curves (embedding degree 1 or 2),
    # we can use the simplified pairing:
    # e(P, Q) = (-1)^n * (y_P * x_Q - x_P * y_Q)^((p-1)/n) mod p

    if P is None or Q is None:
        return 1

    x_p, y_p = P
    x_q, y_q = Q

    # Compute the "distortion map" for supersingular curves
    # For y^2 = x^3 + b (j=0), the distortion is phi(x,y) = (zeta*x, y)
    # where zeta is a cube root of unity
    # For y^2 = x^3 + ax (j=1728), phi(x,y) = (-x, i*y)

    # Simple approach: use the formula for the reduced Tate pairing
    # This works on supersingular curves with embedding degree 2

    # Try a direct computation based on Miller's algorithm (simplified)
    try:
        # For small n, directly compute
        result = 1
        T = P
        bits = bin(n)[2:]

        for bit in bits[1:]:
            # Doubling step
            if T is not None:
                lam_T = (3 * T[0] ** 2 + ec.a) * pow(2 * T[1], p - 2, p) % p if T[1] != 0 else 0
                g_val = (y_q - T[1] - lam_T * (x_q - T[0])) % p if Q != T else 1
                if g_val == 0:
                    g_val = 1
                result = (result * result * g_val) % p

            T = ec.add(T, T)

            if bit == '1':
                if T is not None and P is not None and T != P:
                    lam_TP = (P[1] - T[1]) * pow((P[0] - T[0]) % p, p - 2, p) % p
                    g_val = (y_q - T[1] - lam_TP * (x_q - T[0])) % p
                    if g_val == 0:
                        g_val = 1
                    result = (result * g_val) % p
                T = ec.add(T, P)

        # Final exponentiation
        exp = (p - 1) // n if n > 0 and (p - 1) % n == 0 else p - 1
        result = pow(result, exp, p)
        return result

    except (ZeroDivisionError, ValueError):
        return 1


def mov_attack(ec, G, Q, n, emb_degree):
    """MOV attack: reduce ECDLP to FFDLP via Weil pairing.

    1. Find a linearly independent point T of order n
    2. Compute alpha = e(G, T) in F_{p^k}*
    3. Compute beta = e(Q, T) in F_{p^k}*
    4. Solve DLP in F_{p^k}*: find x such that alpha^x = beta
    5. Then Q = xG
    """
    p = ec.p

    if emb_degree > 6:
        return None, "embedding degree too large"

    if emb_degree == 1:
        # The pairing maps into F_p*
        # Find a second point T of order n that's not a multiple of G
        if ec._points is None:
            ec._enumerate()

        for pt in ec._points[1:]:
            if pt == G:
                continue
            if ec.multiply(pt, n) is not None:
                continue

            # Compute pairings
            alpha = simplified_weil_pairing(ec, G, pt, n)
            beta = simplified_weil_pairing(ec, Q, pt, n)

            if alpha <= 1 or beta <= 1:
                continue

            # Solve DLP in F_p*
            x = baby_step_giant_step_ff(alpha, beta, p, n)
            if x is not None and ec.multiply(G, x) == Q:
                return x, "success"

    return None, "pairing computation failed"


def main():
    print()
    print("=" * 78)
    print("  MOV PAIRING ATTACK")
    print("  Reducing ECDLP to finite field DLP via Weil pairing")
    print("=" * 78)

    # Test various curves and find their embedding degrees
    print(f"\n  EMBEDDING DEGREE ANALYSIS")
    print(f"  {'Curve':20s}  {'p':>8s}  {'|E|':>8s}  {'k':>6s}  {'Vulnerable?':>12s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*12}")

    # Test curves: y^2 = x^3 + 7 (secp256k1 family)
    test_primes = [11, 23, 47, 97, 199, 401, 797, 1601, 3203]
    for p_val in test_primes:
        ec = SmallEC(p_val, 0, 7)
        N = ec.order
        n = N - 1  # subgroup order (for simplicity)
        if n <= 1:
            continue

        # Find largest prime factor of group order
        factors = []
        temp = N
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                while temp % d == 0:
                    temp //= d
                factors.append(d)
            d += 1
        if temp > 1:
            factors.append(temp)
        largest_prime = max(factors) if factors else N

        k = find_embedding_degree(p_val, largest_prime)
        vulnerable = "YES" if k is not None and k <= 6 else "NO"
        k_str = str(k) if k is not None else ">1000"

        print(f"  {'y^2=x^3+7/F_'+str(p_val):20s}  {p_val:>8d}  {N:>8d}  {k_str:>6s}  {vulnerable:>12s}")

    # Test other curve families that might be vulnerable
    print(f"\n  SUPERSINGULAR CURVES (known to have small embedding degree)")
    print(f"  {'Curve':25s}  {'p':>8s}  {'|E|':>8s}  {'k':>6s}  {'Vulnerable?':>12s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*12}")

    # Supersingular: y^2 = x^3 + x over F_p where p = 3 mod 4
    # These have |E| = p + 1 and embedding degree 2
    ss_primes = [7, 11, 23, 43, 67, 83, 107, 163, 227, 283, 443, 563, 787, 1103]
    for p_val in ss_primes:
        if p_val % 4 != 3:
            continue
        ec = SmallEC(p_val, 1, 0)
        N = ec.order

        factors = []
        temp = N
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                while temp % d == 0:
                    temp //= d
                factors.append(d)
            d += 1
        if temp > 1:
            factors.append(temp)
        largest_prime = max(factors) if factors else N

        k = find_embedding_degree(p_val, largest_prime)
        k_str = str(k) if k is not None else ">1000"
        vulnerable = "YES" if k is not None and k <= 6 else "NO"

        print(f"  {'y^2=x^3+x/F_'+str(p_val):25s}  {p_val:>8d}  {N:>8d}  {k_str:>6s}  {vulnerable:>12s}")

    # Attempt MOV attack on vulnerable curves
    print(f"\n\n{'='*78}")
    print(f"  MOV ATTACK ATTEMPTS")
    print(f"{'='*78}")

    # Find and attack supersingular curves with small embedding degree
    attack_results = []

    for p_val in ss_primes:
        if p_val % 4 != 3:
            continue
        ec = SmallEC(p_val, 1, 0)
        N = ec.order
        G = ec.generator

        if G is None or N <= 2:
            continue

        factors = []
        temp = N
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                while temp % d == 0:
                    temp //= d
                factors.append(d)
            d += 1
        if temp > 1:
            factors.append(temp)
        largest_prime = max(factors) if factors else N

        k = find_embedding_degree(p_val, largest_prime)
        if k is None or k > 6:
            continue

        # Random target
        k_target = secrets.randbelow(N - 1) + 1
        Q = ec.multiply(G, k_target)

        t0 = time.time()
        result, status = mov_attack(ec, G, Q, N, k)
        dt = (time.time() - t0) * 1000

        success = result is not None and result == k_target
        print(f"  y^2=x^3+x/F_{p_val}: |E|={N}, k={k}, target={k_target}")
        print(f"    Result: {status}, found={result}, correct={success} ({dt:.1f}ms)")

        attack_results.append({
            "curve": f"y^2=x^3+x/F_{p_val}",
            "p": p_val, "order": N, "emb_degree": k,
            "success": success,
        })

    # Summary
    print(f"\n\n{'='*78}")
    print(f"  SUMMARY: MOV PAIRING ATTACK")
    print(f"{'='*78}")

    n_attacks = len(attack_results)
    n_success = sum(1 for r in attack_results if r["success"])

    print(f"\n  Attacks attempted: {n_attacks}")
    print(f"  Successful: {n_success}")

    print(f"\n  Key insights:")
    print(f"  1. secp256k1 (y^2 = x^3 + 7): embedding degree is HUGE")
    print(f"     -> MOV attack is completely infeasible")
    print(f"     -> This is by design: Koblitz curves are chosen to have large k")
    print(f"")
    print(f"  2. Supersingular curves (y^2 = x^3 + x): embedding degree = 2")
    print(f"     -> MOV attack reduces to F_p^2 DLP")
    print(f"     -> Vulnerable! These curves should NEVER be used for crypto")
    print(f"")
    print(f"  3. The Weil/Tate pairing is the mathematical tool that breaks weak curves")
    print(f"     -> Standard curves (NIST P-256, secp256k1) are immune")
    print(f"     -> Only affects anomalous/supersingular curves with small k")
    print(f"")
    print(f"  4. For secp256k1 specifically:")
    print(f"     -> Embedding degree > 10^70 (essentially p itself)")
    print(f"     -> No pairing-based attack can work")
    print(f"     -> The ONLY known efficient attack is Shor's algorithm")
    print("=" * 78)


if __name__ == "__main__":
    main()
