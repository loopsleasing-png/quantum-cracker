"""Smart's Anomalous Curve Attack.

When |E(F_p)| = p (the curve is "anomalous"), the ECDLP can be
solved in O(log p) time -- essentially LINEAR. This uses p-adic
lifting (Hensel's lemma) to "lift" the curve to Q_p and solve
the DLP there using the p-adic logarithm.

This is a devastating attack on anomalous curves. It was discovered
independently by Smart, Satoh-Araki, and Semaev in 1997-1999.

For secp256k1: |E| = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
              p  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
              |E| != p, so secp256k1 is NOT anomalous. Attack doesn't apply.

But we demonstrate it works on anomalous curves we construct.
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

    def neg(self, P):
        if P is None: return None
        return (P[0], (self.p - P[1]) % self.p)


def find_anomalous_curves(max_p=5000):
    """Find anomalous curves E(F_p) where |E| = p."""
    anomalous = []
    for p in range(5, max_p):
        # Check if p is prime
        if p < 2:
            continue
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if not is_prime:
            continue

        # Try curves y^2 = x^3 + ax + b
        for a in range(0, min(p, 20)):
            for b in range(0, min(p, 20)):
                # Check discriminant: 4a^3 + 27b^2 != 0 mod p
                disc = (4 * a * a * a + 27 * b * b) % p
                if disc == 0:
                    continue

                ec = SmallEC(p, a, b)
                if ec.order == p:
                    anomalous.append((p, a, b, ec))
                    break  # one per prime is enough
            else:
                continue
            break

    return anomalous


def smart_attack(ec, G, Q):
    """Smart's attack on anomalous curves.

    Uses the p-adic elliptic logarithm.

    For an anomalous curve E/F_p (where |E(F_p)| = p):
    1. Lift E, G, Q to E~ over Z/p^2Z
    2. Compute p*G~ and p*Q~ (these are in the kernel of reduction)
    3. The p-adic logarithm of p*G~ and p*Q~ in the formal group
       gives us the discrete log.

    Specifically: k = log_p(Q~) / log_p(G~) mod p
    where log_p is computed via the formula on the formal group.
    """
    p = ec.p
    if ec.order != p:
        return None, "not anomalous"

    # Step 1: Lift the curve to Z/p^2
    # We need to find a', b' such that E~: y^2 = x^3 + a'x + b' over Z/p^2
    # and E~ reduces to E mod p.
    a_lift = ec.a  # works in Z/p^2
    b_lift = ec.b

    # Step 2: Lift the points G and Q to E~(Z/p^2)
    # For a point (x, y) on E, find y_lift such that
    # y_lift^2 ≡ x^3 + a*x + b (mod p^2) and y_lift ≡ y (mod p)
    p2 = p * p

    def lift_point(pt):
        if pt is None:
            return None
        x, y = pt
        # Use Hensel's lemma: if f(y) = y^2 - (x^3 + ax + b) ≡ 0 mod p
        # and f'(y) = 2y ≢ 0 mod p, then we can lift to mod p^2
        rhs = (x * x * x + a_lift * x + b_lift) % p2
        y_sq = (y * y) % p2
        # We need y_lift such that y_lift^2 ≡ rhs mod p^2
        # y_lift = y + t*p where t = (rhs - y^2) / (2*y*p) mod p
        diff = (rhs - y_sq) % p2
        if diff % p != 0:
            return None  # shouldn't happen
        t_numer = diff // p
        try:
            inv_2y = pow(2 * y, -1, p)
        except (ValueError, ZeroDivisionError):
            return None
        t = (t_numer * inv_2y) % p
        y_lift = (y + t * p) % p2
        return (x, y_lift)

    def ec_add_mod(P, Q, mod):
        """EC addition modulo mod."""
        if P is None: return Q
        if Q is None: return P
        x1, y1 = P
        x2, y2 = Q
        if x1 % p == x2 % p and y1 % p == (p - y2 % p) % p:
            return None
        if x1 % p == x2 % p and y1 % p == y2 % p:
            if y1 % p == 0:
                return None
            num = (3 * x1 * x1 + a_lift) % mod
            den = (2 * y1) % mod
            try:
                den_inv = pow(den, -1, mod)
            except (ValueError, ZeroDivisionError):
                return None
            lam = (num * den_inv) % mod
        else:
            num = (y2 - y1) % mod
            den = (x2 - x1) % mod
            try:
                den_inv = pow(den, -1, mod)
            except (ValueError, ZeroDivisionError):
                return None
            lam = (num * den_inv) % mod
        x3 = (lam * lam - x1 - x2) % mod
        y3 = (lam * (x1 - x3) - y1) % mod
        return (x3, y3)

    def ec_mult_mod(P, k, mod):
        """EC scalar multiplication modulo mod."""
        if k == 0 or P is None:
            return None
        result = None
        addend = P
        while k:
            if k & 1:
                result = ec_add_mod(result, addend, mod)
            addend = ec_add_mod(addend, addend, mod)
            k >>= 1
        return result

    G_lift = lift_point(G)
    Q_lift = lift_point(Q)

    if G_lift is None or Q_lift is None:
        return None, "lift failed"

    # Step 3: Compute p*G~ and p*Q~ modulo p^2
    pG = ec_mult_mod(G_lift, p, p2)
    pQ = ec_mult_mod(Q_lift, p, p2)

    if pG is None or pQ is None:
        return None, "p-mult gave infinity"

    # Step 4: Extract the p-adic logarithm
    # For a point (x, y) in the kernel of reduction (i.e., x ≡ 0 mod p, sort of),
    # the formal group logarithm is: log(P) = -x/y mod p
    # Actually for the kernel: if pP = (x', y') with x', y' meaningful mod p^2,
    # then log_p(P) = (x'/p) / (y'/p) ... this needs care.

    # The p-adic log on the formal group:
    # For P = (x, y) with v_p(x) = -2, v_p(y) = -3 in the formal group,
    # the parameter t = -x/y and log(P) = t + ... (higher order terms)

    # Simplified: for points in ker(reduction), the map is
    # phi: ker -> Z/pZ defined by phi(x, y) = -(x/y) mod p
    # and the DLP is k = phi(pQ) / phi(pG) mod p

    x_G, y_G = pG
    x_Q, y_Q = pQ

    # The points pG, pQ should be in the kernel of reduction mod p
    # Their coordinates should satisfy: x ≡ 0 mod p (approximately)
    # The log is: -x/y mod p^2, then divide by p to get mod p

    try:
        log_G = (-x_G * pow(y_G, -1, p2)) % p2
        log_Q = (-x_Q * pow(y_Q, -1, p2)) % p2
    except (ValueError, ZeroDivisionError):
        return None, "log computation failed"

    # These should be multiples of p. Divide by p.
    if log_G % p != 0 or log_Q % p != 0:
        # Try the alternative formula
        log_G = log_G % p2
        log_Q = log_Q % p2
        if log_G == 0:
            return None, "log_G = 0"

    # If they're multiples of p:
    log_G_red = log_G // p if log_G % p == 0 else log_G % p
    log_Q_red = log_Q // p if log_Q % p == 0 else log_Q % p

    if log_G_red == 0:
        return None, "log_G_red = 0"

    try:
        k = (log_Q_red * pow(log_G_red, -1, p)) % p
    except (ValueError, ZeroDivisionError):
        return None, "inverse failed"

    # Verify
    if ec.multiply(G, k) == Q:
        return k, "success"

    # Try p - k
    k2 = (p - k) % p
    if ec.multiply(G, k2) == Q:
        return k2, "success"

    return None, f"verification failed (got {k})"


def main():
    print()
    print("=" * 78)
    print("  SMART'S ANOMALOUS CURVE ATTACK")
    print("  Linear-time DLP on anomalous curves (|E| = p)")
    print("=" * 78)

    # Find anomalous curves
    print(f"\n  Searching for anomalous curves (p < 5000)...")
    t0 = time.time()
    anomalous = find_anomalous_curves(5000)
    dt = time.time() - t0
    print(f"  Found {len(anomalous)} anomalous curves in {dt:.1f}s")

    if not anomalous:
        print(f"  No anomalous curves found! (This is expected for random curves)")
        print(f"  Anomalous curves are VERY rare -- that's why they're not a general threat.")
    else:
        print(f"\n  {'Curve':25s}  {'p':>8s}  {'|E|':>8s}  {'Generator'}")
        print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*20}")
        for p, a, b, ec in anomalous[:15]:
            G = ec.generator
            print(f"  {'y^2=x^3+'+str(a)+'x+'+str(b)+'/F_'+str(p):25s}  {p:>8d}  {ec.order:>8d}  {G}")

    # Attack anomalous curves
    print(f"\n\n{'='*78}")
    print(f"  SMART'S ATTACK ATTEMPTS")
    print(f"{'='*78}")

    total = 0
    success = 0

    for p, a, b, ec in anomalous:
        G = ec.generator
        if G is None:
            continue

        # Test on multiple targets
        for trial in range(min(5, p - 1)):
            k_target = secrets.randbelow(p - 1) + 1
            Q = ec.multiply(G, k_target)

            t0 = time.time()
            k_found, status = smart_attack(ec, G, Q)
            dt = (time.time() - t0) * 1000

            correct = k_found is not None and k_found == k_target
            total += 1
            if correct:
                success += 1

            if trial < 2 or correct:
                print(f"  p={p:5d}: target={k_target:5d}, found={k_found}, "
                      f"status={status}, correct={correct} ({dt:.1f}ms)")

    print(f"\n  Results: {success}/{total} attacks succeeded")

    # ================================================================
    # secp256k1 ANOMALY CHECK
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  secp256k1 ANOMALY CHECK")
    print(f"{'='*78}")

    secp256k1_p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    secp256k1_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    print(f"\n  secp256k1 parameters:")
    print(f"    p (field):    {secp256k1_p}")
    print(f"    n (order):    {secp256k1_n}")
    print(f"    p - n:        {secp256k1_p - secp256k1_n}")
    print(f"    p == n?       {secp256k1_p == secp256k1_n}")

    trace = secp256k1_p + 1 - secp256k1_n
    print(f"    Frobenius trace t = p + 1 - n = {trace}")
    print(f"    Anomalous requires t = 1, actual t = {trace}")
    print(f"\n    VERDICT: secp256k1 is NOT anomalous (t = {trace} != 1)")
    print(f"    Smart's attack does NOT apply.")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"""
  Smart's attack:
  - Solves ECDLP in O(log p) time on ANOMALOUS curves (|E| = p, trace t = 1)
  - Uses p-adic lifting + formal group logarithm
  - Devastating when applicable: linear-time key recovery

  For secp256k1:
  - Frobenius trace t = {trace} (NOT 1)
  - |E| = {secp256k1_n} != p = {secp256k1_p}
  - Smart's attack is COMPLETELY inapplicable
  - This was verified during curve selection (Certicom, 2000)

  Anomalous curves are extremely rare and are screened out
  during standardization. No standard curve is anomalous.
    """)
    print("=" * 78)


if __name__ == "__main__":
    main()
