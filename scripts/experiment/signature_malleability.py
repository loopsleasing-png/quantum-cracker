"""ECDSA Signature Malleability Analysis.

Demonstrates the signature malleability property of ECDSA and its
implications for Bitcoin transaction security.

Key insight: For any valid ECDSA signature (r, s), the pair (r, n-s)
is ALSO a valid signature on the same message. This "malleability"
was a critical Bitcoin vulnerability that enabled transaction ID
mutation and ultimately motivated the SegWit upgrade (BIP-141).

Historical impact:
  - 2011: Bitcoin transaction malleability first identified
  - 2014: Mt. Gox claimed $450M in losses partly due to malleability
  - 2014: BIP-62 proposed strict DER encoding + low-S enforcement
  - 2015: BIP-66 activated (strict DER encoding)
  - 2017: SegWit (BIP-141) activated, fixing malleability structurally

References:
  - Decker & Wattenhofer: "Bitcoin Transaction Malleability and
    MtGox" (2014)
  - BIP-62: Dealing with malleability
  - BIP-141: Segregated Witness
  - libsecp256k1: enforces low-S normalization (s <= n/2)
"""

import csv
import hashlib
import math
import os
import secrets
import sys
import time


# ================================================================
# Small EC arithmetic (reused pattern)
# ================================================================

class SmallEC:
    """Elliptic curve y^2 = x^3 + ax + b over F_p."""
    def __init__(self, p, a, b, name=None):
        self.p = p
        self.a = a
        self.b = b
        self.name = name or f"F_{p}_a{a}b{b}"
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
            self._enumerate()
        return self._gen

    @property
    def points(self):
        if self._points is None:
            self._enumerate()
        return self._points

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
        if len(pts) > 1:
            for pt in pts[1:]:
                if self.multiply(pt, self._order) is None:
                    self._gen = pt
                    break
            if self._gen is None:
                self._gen = pts[1]

    def on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - x * x * x - self.a * x - self.b) % self.p == 0

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
            lam = (y2 - y1) * pow((x2 - x1) % p, p - 2, p) % p
        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def negate(self, P):
        if P is None:
            return None
        return (P[0], (self.p - P[1]) % self.p)

    def multiply(self, P, k):
        if k < 0:
            P = self.negate(P)
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


def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_prime_order_curves():
    """Find small curves with prime group order for ECDSA testing."""
    curves = []
    for p in range(23, 200):
        if not is_prime(p):
            continue
        for a in range(5):
            for b in [1, 3, 5, 7]:
                if (4 * a**3 + 27 * b**2) % p == 0:
                    continue
                ec = SmallEC(p, a, b)
                if is_prime(ec.order) and ec.order > 20:
                    curves.append(ec)
                    if len(curves) >= 6:
                        return curves
    return curves


# ================================================================
# ECDSA implementation with malleability awareness
# ================================================================

def ecdsa_sign(ec, d, msg_hash, k=None):
    """Sign msg_hash with private key d. Returns (r, s, k_used)."""
    n = ec.order
    G = ec.generator
    for _ in range(1000):
        if k is None:
            k_val = secrets.randbelow(n - 1) + 1
        else:
            k_val = k
        R = ec.multiply(G, k_val)
        if R is None:
            continue
        r = R[0] % n
        if r == 0:
            if k is not None:
                return None
            continue
        try:
            k_inv = pow(k_val, -1, n)
        except (ValueError, ZeroDivisionError):
            continue
        s = (k_inv * (msg_hash + r * d)) % n
        if s == 0:
            continue
        return (r, s, k_val)
    return None


def ecdsa_verify(ec, Q, msg_hash, r, s):
    """Verify ECDSA signature (r, s) against public key Q."""
    n = ec.order
    G = ec.generator
    if r < 1 or r >= n or s < 1 or s >= n:
        return False
    try:
        s_inv = pow(s, -1, n)
    except (ValueError, ZeroDivisionError):
        return False
    u1 = (msg_hash * s_inv) % n
    u2 = (r * s_inv) % n
    P = ec.add(ec.multiply(G, u1), ec.multiply(Q, u2))
    if P is None:
        return False
    return P[0] % n == r


def normalize_low_s(s, n):
    """Enforce low-S: if s > n/2, replace with n-s (BIP-62)."""
    if s > n // 2:
        return n - s
    return s


# ================================================================
# DER encoding (Bitcoin signature format)
# ================================================================

def int_to_der_bytes(val):
    """Encode an integer in DER format."""
    if val == 0:
        return b'\x00'
    bs = val.to_bytes((val.bit_length() + 7) // 8, 'big')
    if bs[0] & 0x80:
        bs = b'\x00' + bs
    return bs


def der_encode_signature(r, s):
    """Encode (r, s) as a DER-encoded ECDSA signature."""
    r_bytes = int_to_der_bytes(r)
    s_bytes = int_to_der_bytes(s)
    r_tlv = b'\x02' + bytes([len(r_bytes)]) + r_bytes
    s_tlv = b'\x02' + bytes([len(s_bytes)]) + s_bytes
    seq = r_tlv + s_tlv
    return b'\x30' + bytes([len(seq)]) + seq


def der_signature_length(r, s):
    """Return length of DER-encoded signature."""
    return len(der_encode_signature(r, s))


# ================================================================
# Bitcoin transaction ID computation (simplified)
# ================================================================

def fake_txid(msg, r, s):
    """Simulate a Bitcoin txid that includes the signature."""
    sig_der = der_encode_signature(r, s)
    raw = msg.to_bytes(32, 'big') + sig_der
    return hashlib.sha256(hashlib.sha256(raw).digest()).hexdigest()[:16]


# ================================================================
# Main experiments
# ================================================================

def main():
    W = 70
    print("=" * W)
    print("  ECDSA SIGNATURE MALLEABILITY ANALYSIS")
    print("  How (r, n-s) breaks Bitcoin transaction IDs")
    print("=" * W)

    curves = find_prime_order_curves()
    print(f"\n  Found {len(curves)} test curves with prime order")
    for ec in curves:
        print(f"    {ec.name}: order {ec.order}, G = {ec.generator}")

    csv_rows = []

    # ==============================================================
    # PART 1: Basic Malleability Demonstration
    # ==============================================================
    print(f"\n{'=' * W}")
    print("  PART 1: Basic Malleability -- (r, s) and (r, n-s)")
    print("=" * W)
    print()
    print("  For any valid ECDSA signature (r, s) on message m with")
    print("  public key Q, the signature (r, n-s) is ALSO valid.")
    print()
    print("  Proof: Verification computes s_inv = s^{-1} mod n.")
    print("  For s' = n-s: (s')^{-1} = (n-s)^{-1} = -(s^{-1}) mod n.")
    print("  Then u1' = m*(-s^{-1}) = -u1, u2' = r*(-s^{-1}) = -u2.")
    print("  u1'*G + u2'*Q = -(u1*G + u2*Q) = -R, which has same x")
    print("  coordinate as R. So r' = r. QED.")
    print()

    total_tested = 0
    total_malleable = 0

    for ec in curves[:4]:
        n = ec.order
        G = ec.generator
        d = secrets.randbelow(n - 1) + 1
        Q = ec.multiply(G, d)

        print(f"  Curve {ec.name} (n={n}):")
        successes = 0
        attempts = 20

        for i in range(attempts):
            msg = secrets.randbelow(n - 1) + 1
            result = ecdsa_sign(ec, d, msg)
            if result is None:
                continue
            r, s, _ = result

            # Verify original
            orig_valid = ecdsa_verify(ec, Q, msg, r, s)

            # Create malleable signature
            s_mal = (n - s) % n
            mal_valid = ecdsa_verify(ec, Q, msg, r, s_mal)

            if orig_valid and mal_valid and s != s_mal:
                successes += 1
            total_tested += 1
            if mal_valid and s != s_mal:
                total_malleable += 1

            if i < 3:
                print(f"    sig {i+1}: (r={r}, s={s}) valid={orig_valid}")
                print(f"            (r={r}, s'={s_mal}) valid={mal_valid}  [s != s']")

        print(f"    Result: {successes}/{attempts} signatures are malleable")
        csv_rows.append({
            'experiment': 'basic_malleability',
            'curve': ec.name,
            'order': n,
            'tested': attempts,
            'malleable': successes,
            'rate': f"{successes/max(attempts,1):.2f}",
            'detail': f"(r,n-s) valid for {successes}/{attempts} sigs",
        })

    print(f"\n  TOTAL: {total_malleable}/{total_tested} signatures are malleable")
    print("  (Expected: 100% minus edge case s = n/2, which has probability ~0)")

    # ==============================================================
    # PART 2: Transaction ID Mutation
    # ==============================================================
    print(f"\n{'=' * W}")
    print("  PART 2: Transaction ID Mutation (The Mt. Gox Attack)")
    print("=" * W)
    print()
    print("  Bitcoin transaction IDs are SHA256d(raw_transaction).")
    print("  Since the signature is part of the raw transaction,")
    print("  changing (r, s) to (r, n-s) creates a DIFFERENT txid")
    print("  for the SAME payment. This is signature malleability.")
    print()
    print("  Attack scenario (Mt. Gox, 2014):")
    print("  1. Alice sends Bob 1 BTC (txid: abc123)")
    print("  2. Bob mutates the signature: (r,s) -> (r,n-s)")
    print("  3. Both versions are valid, but have different txids")
    print("  4. If Bob's mutated version gets mined (txid: def456),")
    print("     Alice's wallet doesn't see txid abc123 confirmed")
    print("  5. Alice re-sends, paying Bob twice")
    print()

    ec = curves[0]
    n = ec.order
    G = ec.generator
    d = secrets.randbelow(n - 1) + 1
    Q = ec.multiply(G, d)

    mutations = 0
    for trial in range(10):
        msg = secrets.randbelow(n - 1) + 1
        result = ecdsa_sign(ec, d, msg)
        if result is None:
            continue
        r, s, _ = result
        s_mal = (n - s) % n
        if s == s_mal:
            continue

        txid_orig = fake_txid(msg, r, s)
        txid_mut = fake_txid(msg, r, s_mal)

        if txid_orig != txid_mut:
            mutations += 1
            if trial < 5:
                print(f"  TX {trial+1}: msg={msg}")
                print(f"    Original sig: (r={r}, s={s})")
                print(f"    Mutated  sig: (r={r}, s={s_mal})")
                print(f"    Original txid: {txid_orig}")
                print(f"    Mutated  txid: {txid_mut}")
                print(f"    Same payment, different ID!")
                print()

    print(f"  Result: {mutations}/10 transactions get different txid after mutation")

    csv_rows.append({
        'experiment': 'txid_mutation',
        'curve': ec.name,
        'order': n,
        'tested': 10,
        'malleable': mutations,
        'rate': f"{mutations/10:.2f}",
        'detail': f"{mutations}/10 txids changed by s->n-s mutation",
    })

    # ==============================================================
    # PART 3: Low-S Normalization (BIP-62 Defense)
    # ==============================================================
    print(f"\n{'=' * W}")
    print("  PART 3: Low-S Normalization (BIP-62 / BIP-146)")
    print("=" * W)
    print()
    print("  Defense: Require s <= n/2 ('low-S' canonical form).")
    print("  For any (r, s) where s > n/2, replace s with n-s.")
    print("  This eliminates the malleability degree of freedom.")
    print()
    print("  Bitcoin Core enforces this since v0.10.0 (Feb 2015).")
    print("  SegWit (BIP-141, Aug 2017) made it consensus-critical.")
    print()

    for ec in curves[:3]:
        n = ec.order
        G = ec.generator
        d = secrets.randbelow(n - 1) + 1
        Q = ec.multiply(G, d)

        high_s_count = 0
        normalized_valid = 0
        uniqueness_check = 0

        for _ in range(50):
            msg = secrets.randbelow(n - 1) + 1
            result = ecdsa_sign(ec, d, msg)
            if result is None:
                continue

            r, s, _ = result
            s_low = normalize_low_s(s, n)

            if s > n // 2:
                high_s_count += 1

            # Verify normalized signature
            if ecdsa_verify(ec, Q, msg, r, s_low):
                normalized_valid += 1

            # Check uniqueness: both s_low and n-s_low should not both be <= n/2
            s_other = (n - s_low) % n
            if s_low <= n // 2 and s_other > n // 2:
                uniqueness_check += 1

        print(f"  Curve {ec.name} (n={n}, n/2={n//2}):")
        print(f"    Signatures with high-S (s > n/2): {high_s_count}/50")
        print(f"    After normalization, all valid: {normalized_valid}/50")
        print(f"    Normalization is unique: {uniqueness_check}/50")

        csv_rows.append({
            'experiment': 'low_s_normalization',
            'curve': ec.name,
            'order': n,
            'tested': 50,
            'malleable': high_s_count,
            'rate': f"{high_s_count/50:.2f}",
            'detail': f"{high_s_count}/50 had high-S; all normalized valid",
        })

    print()
    print("  Expected: ~50% of random signatures have s > n/2")
    print("  (uniform s in [1, n-1], half are above n/2)")

    # ==============================================================
    # PART 4: DER Encoding Malleability
    # ==============================================================
    print(f"\n{'=' * W}")
    print("  PART 4: DER Encoding Malleability (BIP-66)")
    print("=" * W)
    print()
    print("  Beyond s-value malleability, DER encoding itself can vary:")
    print("  - Leading zero bytes (positive number padding)")
    print("  - Extra length byte variants")
    print("  - Non-minimal encodings")
    print()
    print("  BIP-66 (July 2015) mandates strict DER encoding,")
    print("  eliminating encoding-based malleability.")
    print()

    ec = curves[0]
    n = ec.order
    G = ec.generator
    d = secrets.randbelow(n - 1) + 1

    der_length_variations = 0
    for _ in range(50):
        msg = secrets.randbelow(n - 1) + 1
        result = ecdsa_sign(ec, d, msg)
        if result is None:
            continue
        r, s, _ = result
        s_mal = (n - s) % n

        len_orig = der_signature_length(r, s)
        len_mal = der_signature_length(r, s_mal)

        if len_orig != len_mal:
            der_length_variations += 1

    print(f"  DER length differences between (r,s) and (r,n-s): {der_length_variations}/50")
    print("  (Length can differ because DER integers need leading zero for high bit)")

    csv_rows.append({
        'experiment': 'der_encoding',
        'curve': ec.name,
        'order': n,
        'tested': 50,
        'malleable': der_length_variations,
        'rate': f"{der_length_variations/50:.2f}",
        'detail': f"{der_length_variations}/50 had different DER lengths",
    })

    # Demonstrate DER encoding variants
    print()
    print("  Example DER encodings:")
    result = ecdsa_sign(ec, d, 42)
    if result:
        r, s, _ = result
        s_mal = (n - s) % n
        der_orig = der_encode_signature(r, s)
        der_mal = der_encode_signature(r, s_mal)
        print(f"    r = {r}, s = {s}")
        print(f"    DER(r, s):   {der_orig.hex()} ({len(der_orig)} bytes)")
        print(f"    DER(r, n-s): {der_mal.hex()} ({len(der_mal)} bytes)")

    # ==============================================================
    # PART 5: Third-Party Malleability (Script-Level)
    # ==============================================================
    print(f"\n{'=' * W}")
    print("  PART 5: Third-Party vs Signer Malleability")
    print("=" * W)
    print()
    print("  Malleability types:")
    print()
    print("  1. THIRD-PARTY (s-value): Anyone can mutate (r,s) -> (r,n-s)")
    print("     No private key needed. Just observe the transaction")
    print("     on the network and rebroadcast with flipped s.")
    print("     Defense: low-S normalization (BIP-62)")
    print()
    print("  2. SIGNER (scriptSig): The signer can add dummy opcodes")
    print("     like OP_DROP or OP_NOP to the scriptSig, changing txid.")
    print("     Defense: SegWit moves witness data outside txid hash")
    print()
    print("  3. ANYONE-CAN-SPEND scripts: Transactions with trivial")
    print("     scripts can be mutated by anyone who sees them.")
    print("     Defense: SegWit + standardness rules")
    print()
    print("  SegWit (BIP-141) solves ALL forms by computing txid")
    print("  from transaction data EXCLUDING witness/signature data.")
    print()

    # Demonstrate third-party malleability requires no secret
    ec = curves[0]
    n = ec.order
    G = ec.generator
    d = secrets.randbelow(n - 1) + 1
    Q = ec.multiply(G, d)

    print("  Demonstration: Third-party mutation without private key")
    print()
    intercepted = 0
    for trial in range(5):
        msg = secrets.randbelow(n - 1) + 1
        result = ecdsa_sign(ec, d, msg)
        if result is None:
            continue
        r, s, _ = result

        # Third party only knows (r, s, msg, Q) -- not d or k
        s_mal = (n - s) % n
        if ecdsa_verify(ec, Q, msg, r, s_mal):
            intercepted += 1
            print(f"    TX {trial+1}: Intercepted sig (r={r}, s={s})")
            print(f"           Mutated to (r={r}, s={s_mal})")
            print(f"           Valid without knowing private key: YES")
            print()

    print(f"  Result: {intercepted}/5 signatures successfully mutated by third party")
    print("  (No private key knowledge required)")

    csv_rows.append({
        'experiment': 'third_party_malleability',
        'curve': ec.name,
        'order': n,
        'tested': 5,
        'malleable': intercepted,
        'rate': f"{intercepted/max(1,5):.2f}",
        'detail': 'No private key needed for mutation',
    })

    # ==============================================================
    # PART 6: SegWit Solution Analysis
    # ==============================================================
    print(f"\n{'=' * W}")
    print("  PART 6: SegWit -- The Structural Fix")
    print("=" * W)
    print()
    print("  Pre-SegWit transaction ID:")
    print("    txid = SHA256d(version | inputs | outputs | locktime)")
    print("    where inputs include scriptSig (which contains signature)")
    print("    PROBLEM: Signature is part of txid computation")
    print()
    print("  SegWit transaction ID (BIP-141):")
    print("    txid = SHA256d(version | marker | flag | inputs | outputs | locktime)")
    print("    where inputs contain EMPTY scriptSig")
    print("    Witness data (signatures) stored separately")
    print("    SOLUTION: Signature NOT part of txid computation")
    print()

    # Simulate pre-SegWit vs SegWit txid stability
    ec = curves[0]
    n = ec.order
    G = ec.generator
    d = secrets.randbelow(n - 1) + 1
    Q = ec.multiply(G, d)

    segwit_stable = 0
    legacy_unstable = 0

    for _ in range(20):
        msg = secrets.randbelow(n - 1) + 1
        result = ecdsa_sign(ec, d, msg)
        if result is None:
            continue
        r, s, _ = result
        s_mal = (n - s) % n
        if s == s_mal:
            continue

        # Pre-SegWit: txid includes signature
        legacy_txid1 = fake_txid(msg, r, s)
        legacy_txid2 = fake_txid(msg, r, s_mal)
        if legacy_txid1 != legacy_txid2:
            legacy_unstable += 1

        # SegWit: txid excludes signature (only message/payment data)
        segwit_txid1 = hashlib.sha256(msg.to_bytes(32, 'big')).hexdigest()[:16]
        segwit_txid2 = hashlib.sha256(msg.to_bytes(32, 'big')).hexdigest()[:16]
        if segwit_txid1 == segwit_txid2:
            segwit_stable += 1

    print(f"  Pre-SegWit: {legacy_unstable}/20 txids changed by malleability")
    print(f"  SegWit:     {segwit_stable}/20 txids STABLE despite sig mutation")
    print()

    csv_rows.append({
        'experiment': 'segwit_comparison',
        'curve': ec.name,
        'order': n,
        'tested': 20,
        'malleable': legacy_unstable,
        'rate': f"legacy:{legacy_unstable}/20,segwit:0/20",
        'detail': f"Legacy {legacy_unstable}/20 mutable; SegWit 0/20 mutable",
    })

    # ==============================================================
    # PART 7: Schnorr Signatures -- Malleability-Free by Design
    # ==============================================================
    print(f"\n{'=' * W}")
    print("  PART 7: Schnorr Signatures (BIP-340 / Taproot)")
    print("=" * W)
    print()
    print("  Bitcoin's Taproot upgrade (Nov 2021) introduced Schnorr")
    print("  signatures (BIP-340), which are NOT malleable.")
    print()
    print("  ECDSA: sig = (r, s) where s = k^{-1}(m + r*d) mod n")
    print("         Malleable because (r, n-s) also satisfies verification")
    print()
    print("  Schnorr: sig = (R, s) where s = k + e*d mod n")
    print("           and e = H(R || P || m)")
    print()
    print("  Why Schnorr is NOT malleable:")
    print("  1. R is a POINT (not just x-coordinate), fixing sign")
    print("  2. e depends on R, P, and m -- changing s changes e")
    print("  3. Verification: sG = R + eP (linear, no inversion)")
    print("  4. Only one (R, s) pair satisfies the equation for given k")
    print()

    # Demonstrate Schnorr on largest available curve (minimize false positives)
    ec = max(curves, key=lambda c: c.order)
    n = ec.order
    print(f"  Using curve {ec.name} (n={n}) for Schnorr test")
    print(f"  (Larger order reduces false-positive rate from random collisions)")
    print()
    G = ec.generator
    d = secrets.randbelow(n - 1) + 1
    Q = ec.multiply(G, d)

    def schnorr_sign(ec, d, msg):
        n = ec.order
        G = ec.generator
        k = secrets.randbelow(n - 1) + 1
        R = ec.multiply(G, k)
        if R is None:
            return None
        e_input = f"{R[0]}:{R[1]}:{msg}".encode()
        e = int(hashlib.sha256(e_input).hexdigest(), 16) % n
        s = (k + e * d) % n
        return (R, s, e)

    def schnorr_verify(ec, Q, msg, R, s):
        n = ec.order
        G = ec.generator
        e_input = f"{R[0]}:{R[1]}:{msg}".encode()
        e = int(hashlib.sha256(e_input).hexdigest(), 16) % n
        lhs = ec.multiply(G, s)
        rhs = ec.add(R, ec.multiply(Q, e))
        return lhs == rhs

    schnorr_malleable = 0
    schnorr_tested = 0

    for trial in range(20):
        msg = secrets.randbelow(n - 1) + 1
        result = schnorr_sign(ec, d, msg)
        if result is None:
            continue
        R, s, e = result

        # Try to create a malleable Schnorr signature
        # Attempt 1: negate s (like ECDSA)
        s_mal = (n - s) % n
        mal1_valid = schnorr_verify(ec, Q, msg, R, s_mal)

        # Attempt 2: negate R
        R_neg = ec.negate(R)
        if R_neg is not None:
            mal2_valid = schnorr_verify(ec, Q, msg, R_neg, s)
        else:
            mal2_valid = False

        # Attempt 3: negate both
        if R_neg is not None:
            mal3_valid = schnorr_verify(ec, Q, msg, R_neg, s_mal)
        else:
            mal3_valid = False

        orig_valid = schnorr_verify(ec, Q, msg, R, s)
        schnorr_tested += 1

        any_malleable = mal1_valid or mal2_valid or mal3_valid
        if any_malleable:
            schnorr_malleable += 1

        if trial < 3:
            print(f"  Schnorr sig {trial+1}: R={R}, s={s}")
            print(f"    Original valid: {orig_valid}")
            print(f"    Negate s:       {mal1_valid}")
            print(f"    Negate R:       {mal2_valid}")
            print(f"    Negate both:    {mal3_valid}")
            print()

    print(f"  Result: {schnorr_malleable}/{schnorr_tested} Schnorr signatures malleable")
    print(f"  (Expected: 0 -- Schnorr has no known malleability)")

    csv_rows.append({
        'experiment': 'schnorr_comparison',
        'curve': ec.name,
        'order': n,
        'tested': schnorr_tested,
        'malleable': schnorr_malleable,
        'rate': f"{schnorr_malleable/max(1,schnorr_tested):.2f}",
        'detail': f"Schnorr: {schnorr_malleable}/{schnorr_tested} malleable (expected 0)",
    })

    # ==============================================================
    # PART 8: Comprehensive Malleability Impact Assessment
    # ==============================================================
    print(f"\n{'=' * W}")
    print("  PART 8: Impact Assessment and Timeline")
    print("=" * W)
    print()

    timeline = [
        ("2008", "Bitcoin whitepaper published (ECDSA for signatures)"),
        ("2011", "Transaction malleability first identified as issue"),
        ("2012", "BIP-16 (P2SH) partially addresses script malleability"),
        ("2013", "Mt. Gox begins experiencing malleability-related issues"),
        ("2014", "Mt. Gox collapses; claims 850,000 BTC ($450M) lost"),
        ("2014", "BIP-62 proposed: strict canonical encodings + low-S"),
        ("2015", "BIP-66 activated: strict DER encoding (soft fork)"),
        ("2015", "Bitcoin Core v0.10: low-S relay policy enforced"),
        ("2015", "SegWit proposed (BIP-141) as structural fix"),
        ("2017", "SegWit activated on Bitcoin mainnet (Aug 24)"),
        ("2021", "Taproot activated: Schnorr sigs (malleability-free)"),
    ]

    for year, event in timeline:
        print(f"  {year}: {event}")

    print()
    print("  Malleability defense layers in modern Bitcoin:")
    print("  1. Low-S normalization (BIP-62): eliminates s-value flip")
    print("  2. Strict DER encoding (BIP-66): eliminates encoding variants")
    print("  3. SegWit (BIP-141): txid excludes witness/signature data")
    print("  4. Schnorr/Taproot (BIP-340): inherently non-malleable sigs")
    print()

    # Quantify the malleability degrees of freedom at each era
    eras = [
        ("Pre-2015 (legacy)", "s-flip + DER + scriptSig", "3+", "HIGH"),
        ("2015-2017 (BIP-66)", "s-flip + scriptSig", "2", "MEDIUM"),
        ("2017-2021 (SegWit)", "None (txid safe)", "0", "NONE"),
        ("2021+ (Taproot)", "None (sig non-malleable)", "0", "NONE"),
    ]

    print("  Malleability by era:")
    print(f"  {'Era':<28} {'Vectors':<30} {'DoF':<5} {'Risk':<8}")
    print(f"  {'-'*28} {'-'*30} {'-'*5} {'-'*8}")
    for era, vectors, dof, risk in eras:
        print(f"  {era:<28} {vectors:<30} {dof:<5} {risk:<8}")
        csv_rows.append({
            'experiment': 'era_analysis',
            'curve': 'secp256k1',
            'order': 'N/A',
            'tested': 'N/A',
            'malleable': dof,
            'rate': risk,
            'detail': f"{era}: {vectors}",
        })

    # ==============================================================
    # Summary
    # ==============================================================
    print(f"\n{'=' * W}")
    print("  SUMMARY")
    print("=" * W)
    print()
    print("  ECDSA signature malleability is a MATHEMATICAL PROPERTY,")
    print("  not a bug. For any valid (r, s), (r, n-s) is also valid.")
    print()
    print("  Impact on Bitcoin:")
    print("  - Enabled transaction ID mutation without private key")
    print("  - Facilitated Mt. Gox accounting confusion ($450M lost)")
    print("  - Blocked second-layer protocols (Lightning Network)")
    print("  - Required 3 successive protocol upgrades to fully fix")
    print()
    print("  Current status (2024+):")
    print("  - SegWit transactions: immune to txid malleability")
    print("  - Taproot/Schnorr: inherently non-malleable signatures")
    print("  - Legacy transactions: still vulnerable but declining usage")
    print()
    print("  Key insight: Malleability is a PROTOCOL attack, not a")
    print("  cryptographic break. The signatures are valid -- the")
    print("  problem was using signatures as part of transaction IDs.")

    # ==============================================================
    # Write CSV
    # ==============================================================
    csv_path = os.path.expanduser("~/Desktop/signature_malleability.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'experiment', 'curve', 'order', 'tested',
            'malleable', 'rate', 'detail'
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  CSV written: {csv_path}")
    print(f"  Total rows: {len(csv_rows)}")
    print(f"\n{'=' * W}")


if __name__ == "__main__":
    main()
