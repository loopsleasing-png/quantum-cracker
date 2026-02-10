"""Invalid Curve Attack (Small Subgroup Attack).

A devastating IMPLEMENTATION attack that exploits missing point validation.

If an ECDH implementation doesn't verify that received points actually
lie on the correct curve, an attacker can send points on a DIFFERENT
(weaker) curve with small subgroup order, and leak private key bits.

This is NOT a mathematical weakness of secp256k1 itself, but a
software vulnerability in implementations that skip validation.

Real-world impact:
  - CVE-2017-8932 (Go crypto/elliptic): invalid curve point accepted
  - CVE-2020-0601 (Windows CryptoAPI): accepted points on wrong curve
  - Multiple TLS implementations vulnerable pre-2015

References:
  - Biehl, Meyer, Muller: "Differential Fault Attacks on Elliptic
    Curve Cryptosystems" (CRYPTO 2000)
  - Jager, Schwenk, Somorovsky: "Practical Invalid Curve Attacks"
    (ESORICS 2017)
"""

import csv
import math
import os
import secrets
import sys
import time

sys.path.insert(0, "src")


class ECurve:
    """Elliptic curve y^2 = x^3 + ax + b over F_p.

    Deliberately allows points NOT on this specific curve
    to demonstrate the invalid curve attack.
    """
    def __init__(self, p, a, b, validate=True):
        self.p = p
        self.a = a
        self.b = b
        self.validate = validate
        self._order = None
        self._points = None

    @property
    def order(self):
        if self._order is None:
            self._enumerate()
        return self._order

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

    def on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - x * x * x - self.a * x - self.b) % self.p == 0

    def add(self, P, Q):
        """Point addition. If validate=True, checks points are on curve."""
        if P is None: return Q
        if Q is None: return P
        if self.validate:
            if not self.on_curve(P):
                raise ValueError(f"Point {P} not on curve!")
            if not self.on_curve(Q):
                raise ValueError(f"Point {Q} not on curve!")
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


def find_point_on_twist(p, a, target_x=None):
    """Find a point on an invalid curve y^2 = x^3 + ax + b'
    for various b' values, looking for small-order subgroups.
    """
    results = []
    for b_prime in range(p):
        if b_prime == 7:  # skip the real curve
            continue
        disc = (4 * a * a * a + 27 * b_prime * b_prime) % p
        if disc == 0:
            continue
        ec_invalid = ECurve(p, a, b_prime, validate=False)
        order = ec_invalid.order
        if order <= 2:
            continue
        factors = factorize(order)
        # Look for small prime factors
        for q in sorted(factors.keys()):
            if q <= 50:  # small subgroup
                cofactor = order // q
                # Find a generator of the subgroup of order q
                for pt in (ec_invalid._points or [])[1:]:
                    sub_pt = ec_invalid.multiply(pt, cofactor)
                    if sub_pt is not None and ec_invalid.multiply(sub_pt, q) is None:
                        results.append({
                            "b_prime": b_prime,
                            "order": order,
                            "subgroup_order": q,
                            "generator": sub_pt,
                            "curve": ec_invalid,
                        })
                        break
    return results


def invalid_curve_attack(p, a, victim_key, max_queries=50):
    """Simulate an invalid curve attack.

    The attacker:
    1. Finds points on invalid curves with small subgroup orders
    2. Sends these to the victim as ECDH public keys
    3. The victim (without validation) computes k * P where k is their private key
    4. The victim returns k * P (or a function of it)
    5. The attacker recovers k mod q for each small subgroup order q
    6. CRT combines the residues to get k

    victim_key: the private key we're trying to recover
    """
    # Step 1: Find invalid curve points with small subgroups
    invalid_points = find_point_on_twist(p, a)

    if not invalid_points:
        return None, 0, "no invalid points found"

    # Collect unique subgroup orders
    seen_orders = set()
    useful_points = []
    for info in invalid_points:
        q = info["subgroup_order"]
        if q not in seen_orders:
            seen_orders.add(q)
            useful_points.append(info)

    residues = []
    moduli = []
    queries = 0

    for info in useful_points:
        if queries >= max_queries:
            break

        q = info["subgroup_order"]
        P_invalid = info["generator"]
        ec_invalid = info["curve"]

        # Step 2: "Send" P_invalid to the victim
        # Step 3: Victim computes k * P_invalid (without validation!)
        # The victim's code doesn't check if P_invalid is on their curve
        victim_ec = ECurve(p, a, info["b_prime"], validate=False)
        result = victim_ec.multiply(P_invalid, victim_key)
        queries += 1

        # Step 4: Attacker receives the result
        # Step 5: Since P_invalid has order q, result = (k mod q) * P_invalid
        # Brute-force search in the small subgroup
        k_mod_q = None
        for d in range(q):
            if victim_ec.multiply(P_invalid, d) == result:
                k_mod_q = d
                break

        if k_mod_q is not None:
            residues.append(k_mod_q)
            moduli.append(q)

    if not residues:
        return None, queries, "no residues recovered"

    # Step 6: CRT to combine
    M = 1
    for m in moduli:
        M *= m

    k_recovered = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        try:
            yi = pow(Mi, -1, m)
        except (ValueError, ZeroDivisionError):
            continue
        k_recovered += r * Mi * yi
    k_recovered = k_recovered % M

    return k_recovered, queries, f"recovered k mod {M} = {k_recovered} (need k mod order for full key)"


def main():
    print()
    print("=" * 78)
    print("  INVALID CURVE ATTACK (Small Subgroup Attack)")
    print("  Exploiting missing point validation in ECDH")
    print("=" * 78)

    csv_rows = []

    # ================================================================
    # PART 1: Find invalid curves with small subgroups
    # ================================================================
    print(f"\n  PART 1: Finding Invalid Curves with Small Subgroups")
    print(f"  {'='*70}")

    test_primes = [101, 251, 503, 1009, 2003]

    for p in test_primes:
        print(f"\n  Field F_{p}:")
        real_ec = ECurve(p, 0, 7)
        print(f"    Real curve y^2 = x^3 + 7: |E| = {real_ec.order}")

        invalid_pts = find_point_on_twist(p, 0)
        subgroup_orders = sorted(set(info["subgroup_order"] for info in invalid_pts))
        print(f"    Invalid curves found: {len(invalid_pts)} points")
        print(f"    Small subgroup orders available: {subgroup_orders[:15]}")

        product = 1
        for q in subgroup_orders:
            product *= q
        print(f"    Product of subgroup orders: {product} "
              f"({'ENOUGH' if product >= real_ec.order else 'NOT ENOUGH'} to recover full key)")

    # ================================================================
    # PART 2: Attack simulation
    # ================================================================
    print(f"\n\n  PART 2: Invalid Curve Attack Simulation")
    print(f"  {'='*70}")

    for p in test_primes:
        real_ec = ECurve(p, 0, 7)
        N = real_ec.order
        if N <= 2:
            continue

        n_success = 0
        n_partial = 0
        n_total = 5

        for trial in range(n_total):
            victim_key = secrets.randbelow(N - 1) + 1

            t0 = time.time()
            k_recovered, queries, status = invalid_curve_attack(p, 0, victim_key)
            dt = (time.time() - t0) * 1000

            if k_recovered is not None:
                # Check if we recovered the full key or partial
                if k_recovered == victim_key:
                    n_success += 1
                    result = "FULL KEY"
                elif k_recovered == victim_key % k_recovered if k_recovered > 0 else False:
                    n_partial += 1
                    result = "PARTIAL"
                else:
                    n_partial += 1
                    result = "PARTIAL"
            else:
                result = "FAILED"

            if trial < 2:
                print(f"  p={p:5d}: key={victim_key:6d}, recovered={k_recovered}, "
                      f"queries={queries}, {result} ({dt:.1f}ms)")

            csv_rows.append({
                "prime": p,
                "order": N,
                "victim_key": victim_key,
                "recovered_key": k_recovered if k_recovered is not None else "",
                "queries": queries,
                "result": result,
                "time_ms": round(dt, 2),
            })

        print(f"  p={p}: {n_success}/{n_total} full, {n_partial}/{n_total} partial")

    # ================================================================
    # PART 3: With vs Without Validation
    # ================================================================
    print(f"\n\n  PART 3: The Fix -- Point Validation")
    print(f"  {'='*70}")

    p = 503
    real_ec_valid = ECurve(p, 0, 7, validate=True)
    real_ec_no_valid = ECurve(p, 0, 7, validate=False)

    # Find an invalid point
    invalid_pts = find_point_on_twist(p, 0)
    if invalid_pts:
        P_invalid = invalid_pts[0]["generator"]
        b_prime = invalid_pts[0]["b_prime"]
        q = invalid_pts[0]["subgroup_order"]

        print(f"\n  Invalid point: {P_invalid} (on y^2 = x^3 + {b_prime}, order {q})")
        print(f"  On real curve? {real_ec_valid.on_curve(P_invalid)}")

        print(f"\n  WITHOUT validation (vulnerable):")
        try:
            result = real_ec_no_valid.multiply(P_invalid, 42)
            print(f"    42 * P_invalid = {result}  -- ATTACK SUCCEEDS")
        except ValueError as e:
            print(f"    Error: {e}")

        print(f"\n  WITH validation (secure):")
        try:
            result = real_ec_valid.multiply(P_invalid, 42)
            print(f"    42 * P_invalid = {result}  -- THIS SHOULD NOT HAPPEN")
        except ValueError as e:
            print(f"    Error: {e}  -- ATTACK BLOCKED!")

    # ================================================================
    # PART 4: secp256k1 Analysis
    # ================================================================
    print(f"\n\n  PART 4: secp256k1 Invalid Curve Risk Assessment")
    print(f"  {'='*70}")

    secp256k1_p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    secp256k1_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    print(f"""
  secp256k1: y^2 = x^3 + 7 over F_p
    p = {secp256k1_p}
    n = {secp256k1_n}
    cofactor h = 1 (group order = subgroup order)

  Twist of secp256k1: y^2 = x^3 + 7 has a quadratic twist with order:
    n_twist = 2*p + 2 - n = {2 * secp256k1_p + 2 - secp256k1_n}

  The twist order factors determine what subgroups an attacker could use.
  For secp256k1's twist, the order has been analyzed:
    n_twist = 2 * 3 * ... (factors not trivially small)

  HOWEVER: this attack is ENTIRELY preventable:
    1. Validate that received points are on the correct curve
    2. Check that points are in the correct subgroup (cofactor multiplication)
    3. Use compressed point format (forces y^2 check during decompression)

  libsecp256k1 does ALL THREE. Bitcoin is immune.

  Real-world CVEs from missing validation:
    - CVE-2017-8932: Go crypto/elliptic invalid point multiplication
    - CVE-2020-0601: Windows CryptoAPI curveball (wrong curve accepted)
    - CVE-2014-0160: Not directly related but similar class of input validation bug
    """)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"""
  Invalid curve attack:
  - Sends points on a DIFFERENT curve to exploit missing validation
  - Small subgroups on invalid curves leak k mod q for each subgroup order q
  - CRT combines partial information to recover the full key
  - Requires O(sum(q_i)) work where q_i are small subgroup orders

  This is an IMPLEMENTATION attack, not a mathematical one:
  - The math of secp256k1 is fine
  - The IMPLEMENTATION must validate inputs
  - One missing check = complete key recovery

  Defense (all implemented in libsecp256k1):
  - Verify y^2 = x^3 + 7 for all received points
  - Verify point is not at infinity
  - Verify point order = n (via cofactor multiplication, trivial for h=1)
  - Use compressed points (decompression inherently validates)

  Bottom line: another attack that DOESN'T apply to Bitcoin
  (libsecp256k1 validates everything), but devastating against
  naive implementations.
    """)
    print("=" * 78)

    # Write CSV
    desktop = os.path.expanduser("~/Desktop")
    csv_path = os.path.join(desktop, "invalid_curve_attack.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"  CSV written to {csv_path}")


if __name__ == "__main__":
    main()
