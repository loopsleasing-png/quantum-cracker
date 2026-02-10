"""Pohlig-Hellman Attack -- Deep Dive.

The Pohlig-Hellman algorithm exploits smooth group orders.
If |E(F_p)| = p1^e1 * p2^e2 * ... * pk^ek, then the DLP
can be solved in time O(sum(ei * (sqrt(pi) + log(pi))))
instead of O(sqrt(N)).

This is WHY secp256k1 was chosen with PRIME group order:
  n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
  n is prime, so Pohlig-Hellman gives zero speedup.

If the order were smooth (many small factors), we could
crack the key trivially. This script demonstrates both cases.

References:
  - Pohlig & Hellman, "An Improved Algorithm for Computing
    Logarithms over GF(p)", IEEE Trans. IT, 1978
  - secp256k1 order primality: verified by Certicom (2000)
"""

import csv
import math
import os
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
        if self._points and len(self._points) > 1:
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


def factorize(n):
    """Simple trial division factorization."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def bsgs(ec, G, Q, n):
    """Baby-step giant-step for DLP in subgroup of order n."""
    m = int(math.isqrt(n)) + 1
    # Baby steps: j*G for j in [0, m)
    baby = {}
    current = None  # identity
    for j in range(m):
        baby[current] = j
        current = ec.add(current, G)

    # Giant steps: Q - i*m*G for i in [0, m)
    mG = ec.multiply(G, m)
    neg_mG = ec.neg(mG)
    current = Q
    for i in range(m):
        if current in baby:
            k = (baby[current] + i * m) % n
            return k
        current = ec.add(current, neg_mG)
    return None


def pohlig_hellman(ec, G, Q, order, factors):
    """Full Pohlig-Hellman algorithm.

    Solves DLP by reducing to subgroups of prime-power order,
    solving each independently, then combining via CRT.
    """
    residues = []
    moduli = []
    total_ops = 0

    for p_i, e_i in factors.items():
        # Compute DLP modulo p_i^e_i
        q = p_i ** e_i
        cofactor = order // q

        # Project to subgroup of order q
        G_sub = ec.multiply(G, cofactor)
        Q_sub = ec.multiply(Q, cofactor)

        if G_sub is None:
            continue

        # For prime power: decompose further
        # k mod p_i^e_i = d_0 + d_1*p_i + d_2*p_i^2 + ...
        k_mod = 0
        Q_temp = Q_sub

        gamma = ec.multiply(G_sub, q // p_i)  # generator of order p_i

        for j in range(e_i):
            # Compute Q_temp * (q / p_i^(j+1))
            exp = q // (p_i ** (j + 1))
            Q_proj = ec.multiply(Q_temp, exp)

            # Solve DLP in subgroup of prime order p_i
            d_j = bsgs(ec, gamma, Q_proj, p_i)
            total_ops += int(math.isqrt(p_i)) + 1

            if d_j is None:
                d_j = 0  # fallback

            k_mod += d_j * (p_i ** j)

            # Update Q_temp
            Q_temp = ec.add(Q_temp, ec.neg(ec.multiply(G_sub, d_j * (p_i ** j))))

        residues.append(k_mod % q)
        moduli.append(q)

    # CRT combination
    if not residues:
        return None, total_ops

    k = crt(residues, moduli)
    return k % order if k is not None else None, total_ops


def crt(residues, moduli):
    """Chinese Remainder Theorem."""
    if not residues:
        return 0
    M = 1
    for m in moduli:
        M *= m
    result = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        try:
            yi = pow(Mi, -1, m)
        except (ValueError, ZeroDivisionError):
            continue
        result += r * Mi * yi
    return result % M


def smoothness_score(n):
    """Measure how smooth a number is.
    Returns the largest prime factor -- smaller = smoother.
    """
    factors = factorize(n)
    if not factors:
        return n
    return max(factors.keys())


def main():
    print()
    print("=" * 78)
    print("  POHLIG-HELLMAN ATTACK -- DEEP DIVE")
    print("  Why smooth-order curves die, and why secp256k1 survives")
    print("=" * 78)

    # ================================================================
    # PART 1: Find curves with smooth vs prime orders
    # ================================================================
    print(f"\n  PART 1: Smooth vs Prime Order Curves")
    print(f"  {'='*70}")

    smooth_curves = []
    prime_curves = []

    for p in range(101, 5000):
        is_prime = all(p % d != 0 for d in range(2, int(p**0.5) + 1))
        if not is_prime or p < 5:
            continue

        ec = SmallEC(p, 0, 7)  # y^2 = x^3 + 7 (same form as secp256k1)
        N = ec.order
        if N <= 2:
            continue

        factors = factorize(N)
        largest_factor = smoothness_score(N)
        n_factors = sum(factors.values())

        if largest_factor < 50 and n_factors >= 3:
            smooth_curves.append((p, N, factors, ec))
        elif len(factors) == 1 and list(factors.values())[0] == 1:
            prime_curves.append((p, N, factors, ec))

    print(f"\n  Found {len(smooth_curves)} smooth-order curves, "
          f"{len(prime_curves)} prime-order curves")

    # Show examples
    print(f"\n  Smooth-order curves (VULNERABLE to Pohlig-Hellman):")
    print(f"  {'p':>8s}  {'|E|':>8s}  {'Factorization':30s}  {'Largest factor':>15s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*30}  {'-'*15}")
    for p, N, factors, ec in smooth_curves[:10]:
        fact_str = " * ".join(f"{pi}^{ei}" if ei > 1 else str(pi)
                              for pi, ei in sorted(factors.items()))
        print(f"  {p:8d}  {N:8d}  {fact_str:30s}  {smoothness_score(N):>15d}")

    print(f"\n  Prime-order curves (IMMUNE to Pohlig-Hellman):")
    print(f"  {'p':>8s}  {'|E|':>8s}  {'Order is prime':>15s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*15}")
    for p, N, factors, ec in prime_curves[:10]:
        print(f"  {p:8d}  {N:8d}  {'YES':>15s}")

    # ================================================================
    # PART 2: Attack smooth-order curves
    # ================================================================
    print(f"\n\n  PART 2: Pohlig-Hellman Attack on Smooth-Order Curves")
    print(f"  {'='*70}")

    csv_rows = []
    n_ph_success = 0
    n_ph_total = 0
    n_bsgs_success = 0

    for p, N, factors, ec in smooth_curves[:20]:
        G = ec.generator
        if G is None:
            continue

        n = N  # group order
        largest = smoothness_score(n)

        for trial in range(3):
            k_target = secrets.randbelow(n - 1) + 1
            Q = ec.multiply(G, k_target)

            # Pohlig-Hellman attack
            t0 = time.time()
            k_ph, ph_ops = pohlig_hellman(ec, G, Q, n, factors)
            dt_ph = (time.time() - t0) * 1000

            # BSGS baseline
            t0 = time.time()
            k_bsgs = bsgs(ec, G, Q, n)
            dt_bsgs = (time.time() - t0) * 1000
            bsgs_ops = int(math.isqrt(n)) + 1

            ph_correct = k_ph is not None and ec.multiply(G, k_ph) == Q
            bsgs_correct = k_bsgs is not None and ec.multiply(G, k_bsgs) == Q

            n_ph_total += 1
            if ph_correct:
                n_ph_success += 1
            if bsgs_correct:
                n_bsgs_success += 1

            speedup = bsgs_ops / max(ph_ops, 1)

            if trial == 0:
                print(f"  p={p:5d}, |E|={n:6d}, largest_factor={largest:4d}: "
                      f"PH={ph_ops:5d} ops ({dt_ph:.1f}ms) "
                      f"BSGS={bsgs_ops:5d} ops ({dt_bsgs:.1f}ms) "
                      f"speedup={speedup:.1f}x  PH_correct={ph_correct}")

            csv_rows.append({
                "prime": p,
                "order": n,
                "largest_factor": largest,
                "n_factors": sum(factors.values()),
                "method": "pohlig_hellman",
                "ops": ph_ops,
                "time_ms": round(dt_ph, 2),
                "correct": ph_correct,
                "speedup_vs_bsgs": round(speedup, 2),
            })

    print(f"\n  Pohlig-Hellman results: {n_ph_success}/{n_ph_total} correct")
    print(f"  BSGS baseline results: {n_bsgs_success}/{n_ph_total} correct")

    # ================================================================
    # PART 3: Attack prime-order curves (should give no speedup)
    # ================================================================
    print(f"\n\n  PART 3: Pohlig-Hellman on Prime-Order Curves (No Speedup)")
    print(f"  {'='*70}")

    for p, N, factors, ec in prime_curves[:5]:
        G = ec.generator
        if G is None:
            continue

        n = N
        k_target = secrets.randbelow(n - 1) + 1
        Q = ec.multiply(G, k_target)

        t0 = time.time()
        k_ph, ph_ops = pohlig_hellman(ec, G, Q, n, factors)
        dt_ph = (time.time() - t0) * 1000

        bsgs_ops = int(math.isqrt(n)) + 1
        speedup = bsgs_ops / max(ph_ops, 1)

        ph_correct = k_ph is not None and ec.multiply(G, k_ph) == Q

        print(f"  p={p:5d}, |E|={n:6d} (PRIME): "
              f"PH={ph_ops:5d} ops, BSGS={bsgs_ops:5d} ops, "
              f"speedup={speedup:.1f}x  correct={ph_correct}")

    # ================================================================
    # PART 4: secp256k1 Analysis
    # ================================================================
    print(f"\n\n  PART 4: secp256k1 Order Analysis")
    print(f"  {'='*70}")

    secp256k1_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    # Check primality using Fermat test + Miller-Rabin
    print(f"\n  secp256k1 group order n:")
    print(f"    n = {secp256k1_n}")
    print(f"    n (hex) = {hex(secp256k1_n)}")
    print(f"    bit length = {secp256k1_n.bit_length()}")

    # Miller-Rabin primality test
    def is_probable_prime(n, k=20):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        for _ in range(k):
            a = secrets.randbelow(n - 3) + 2
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

    is_prime = is_probable_prime(secp256k1_n)
    print(f"\n    Miller-Rabin primality (20 rounds): {'PRIME' if is_prime else 'COMPOSITE'}")
    print(f"    Probability of error: < 4^(-20) = {4**-20:.2e}")

    print(f"\n    Since n is PRIME:")
    print(f"      - Pohlig-Hellman gives ZERO speedup")
    print(f"      - Only subgroup is the full group itself")
    print(f"      - DLP complexity remains O(sqrt(n)) = O(2^128)")

    # What if the order were smooth?
    print(f"\n    HYPOTHETICAL: What if n were 2^256 - 2^32 + something smooth?")
    hypothetical_smooth = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # That's about 614889782588491410 ~ 2^59
    # For 256-bit smooth, we'd need product of primes up to ~50
    print(f"    Product of primes 2..47: {hypothetical_smooth} (~2^{hypothetical_smooth.bit_length()} bits)")
    print(f"    Pohlig-Hellman on smooth 256-bit order: O(sqrt(47)) = ~7 ops per factor")
    print(f"    Total: ~{7 * 15} ops to crack a key (vs 2^128 for prime order)")
    print(f"    That's why smooth orders are CATASTROPHIC and why n must be prime!")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"""
  Pohlig-Hellman attack:
  - Reduces DLP from O(sqrt(N)) to O(sum(sqrt(p_i))) where p_i are prime factors
  - Smooth orders (many small factors): devastating speedup
  - Prime orders: NO speedup at all (single factor = full group)

  secp256k1:
  - Group order n is PRIME (verified by Miller-Rabin, 20 rounds)
  - Pohlig-Hellman is completely inapplicable
  - This is BY DESIGN -- Certicom chose curves with prime order

  On our test curves:
  - Smooth-order: Pohlig-Hellman needed {n_ph_success}/{n_ph_total} solved correctly
  - Speedup over BSGS on smooth curves: significant when largest factor < 50
  - Prime-order: Pohlig-Hellman = BSGS (no speedup, as expected)

  Key insight: the choice of group order is as important as the choice
  of curve equation. A "secure" curve with smooth order is trivially broken.
    """)
    print("=" * 78)

    # Write CSV
    desktop = os.path.expanduser("~/Desktop")
    csv_path = os.path.join(desktop, "pohlig_hellman_analysis.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"  CSV written to {csv_path}")


if __name__ == "__main__":
    main()
