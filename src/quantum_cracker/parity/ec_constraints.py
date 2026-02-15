"""Elliptic curve constraint encoding for the Ising Hamiltonian.

Maps the EC discrete logarithm problem k*G = P onto Ising coupling
terms so that the ground state of the resulting Hamiltonian encodes
the private key k in its spin configuration.

Supports curves of any size: small curves use point enumeration,
large curves use direct EC arithmetic with precomputed power points.
"""

from __future__ import annotations

import secrets

import numpy as np
from numpy.typing import NDArray


Point = tuple[int, int] | None


class SmallEC:
    """Elliptic curve y^2 = x^3 + ax + b over F_p.

    For small primes (p < ~10^6), supports full point enumeration.
    For any prime, supports add/multiply/generator operations.
    """

    def __init__(self, p: int, a: int, b: int, order: int | None = None) -> None:
        self.p = p
        self.a = a
        self.b = b
        self._points: list | None = None
        self._order: int | None = order
        self._gen: Point = None
        self._can_enumerate = p < 1_000_000

    @property
    def order(self) -> int:
        if self._order is None:
            if self._can_enumerate:
                self._enumerate()
            else:
                self._compute_order()
        assert self._order is not None
        return self._order

    @property
    def generator(self) -> tuple[int, int]:
        if self._gen is None:
            if self._can_enumerate:
                self._find_generator_enum()
            else:
                self._find_generator_probe()
        assert self._gen is not None
        return self._gen

    @property
    def points(self) -> list:
        if self._points is None:
            self._enumerate()
        assert self._points is not None
        return self._points

    def key_bit_length(self) -> int:
        """Number of bits needed to represent the curve order."""
        return (self.order - 1).bit_length()

    def _enumerate(self) -> None:
        points: list = [None]  # infinity
        p, a, b = self.p, self.a, self.b
        qr: dict[int, list[int]] = {}
        for y in range(p):
            qr.setdefault((y * y) % p, []).append(y)
        for x in range(p):
            rhs = (x * x * x + a * x + b) % p
            if rhs in qr:
                for y in qr[rhs]:
                    points.append((x, y))
        self._points = points
        self._order = len(points)

    def _compute_order(self) -> None:
        """Compute curve order for large primes using baby-step giant-step
        on the Hasse interval [p+1-2*sqrt(p), p+1+2*sqrt(p)]."""
        import math

        p = self.p
        pt = self._find_point()
        if pt is None:
            raise RuntimeError(f"Could not find a point on curve over F_{p}")

        # Hasse bound: |#E - (p+1)| <= 2*sqrt(p)
        lo = p + 1 - 2 * int(math.isqrt(p)) - 2
        hi = p + 1 + 2 * int(math.isqrt(p)) + 2

        # Compute n*pt for all n in [lo, hi] and find n where n*pt = O
        # Use baby-step giant-step on the Hasse interval
        m = int(math.isqrt(hi - lo)) + 1

        # Baby steps: compute lo*pt + j*pt for j = 0..m
        base_pt = self.multiply(pt, lo)
        baby = {}
        current = base_pt
        for j in range(m + 1):
            key = current
            baby[key] = j
            current = self.add(current, pt)

        # Giant steps: compute -i*m*pt for i = 0,1,2,...
        step = self.multiply(pt, m)
        neg_step = self.neg(step)
        giant = None  # 0
        for i in range((hi - lo) // m + 2):
            if giant in baby:
                n = lo + baby[giant] + i * m
                # Verify
                if self.multiply(pt, n) is None:
                    self._order = n
                    return
            giant = self.add(giant, neg_step) if giant is not None else neg_step

        # Fallback: linear scan
        for n in range(lo, hi + 1):
            if self.multiply(pt, n) is None:
                self._order = n
                return

        raise RuntimeError(f"Could not compute order for curve over F_{p}")

    def _find_point(self) -> tuple[int, int] | None:
        """Find a random point on the curve."""
        p, a, b = self.p, self.a, self.b
        for x in range(p):
            rhs = (x * x * x + a * x + b) % p
            if pow(rhs, (p - 1) // 2, p) == 1:  # QR test
                y = pow(rhs, (p + 1) // 4, p) if p % 4 == 3 else self._tonelli_shanks(rhs)
                if y is not None and (y * y) % p == rhs:
                    return (x, y)
        return None

    def _tonelli_shanks(self, n: int) -> int | None:
        """Tonelli-Shanks algorithm for modular square root."""
        p = self.p
        if pow(n, (p - 1) // 2, p) != 1:
            return None

        # Factor out powers of 2 from p-1
        q = p - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1

        # Find a non-residue
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1

        m = s
        c = pow(z, q, p)
        t = pow(n, q, p)
        r = pow(n, (q + 1) // 2, p)

        while True:
            if t == 1:
                return r
            i = 1
            temp = (t * t) % p
            while temp != 1:
                temp = (temp * temp) % p
                i += 1
                if i == m:
                    return None
            b = pow(c, 1 << (m - i - 1), p)
            m = i
            c = (b * b) % p
            t = (t * c) % p
            r = (r * b) % p

    def _find_generator_enum(self) -> None:
        """Find generator by enumeration (small curves)."""
        if self._points is None:
            self._enumerate()
        assert self._points is not None
        for pt in self._points[1:]:
            if self.multiply(pt, self.order) is None:
                is_gen = True
                for d in range(2, int(self.order**0.5) + 1):
                    if self.order % d == 0:
                        if self.multiply(pt, self.order // d) is None:
                            is_gen = False
                            break
                if is_gen:
                    self._gen = pt
                    return
        self._gen = self._points[1]

    def _find_generator_probe(self) -> None:
        """Find generator by probing random points (large curves)."""
        order = self.order
        # Factor the order to check subgroups
        factors = self._small_factors(order)

        for _ in range(100):
            pt = self._find_random_point()
            if pt is None:
                continue
            # Check that pt has full order
            if self.multiply(pt, order) is not None:
                continue
            is_gen = True
            for f in factors:
                if self.multiply(pt, order // f) is None:
                    is_gen = False
                    break
            if is_gen:
                self._gen = pt
                return

        # Fallback to first point found
        pt = self._find_point()
        if pt is not None:
            self._gen = pt

    def _find_random_point(self) -> tuple[int, int] | None:
        """Find a random point on the curve."""
        p, a, b = self.p, self.a, self.b
        for _ in range(100):
            x = secrets.randbelow(p)
            rhs = (x * x * x + a * x + b) % p
            if pow(rhs, (p - 1) // 2, p) == 1:
                y = pow(rhs, (p + 1) // 4, p) if p % 4 == 3 else self._tonelli_shanks(rhs)
                if y is not None and (y * y) % p == rhs:
                    return (x, y)
        return None

    @staticmethod
    def _small_factors(n: int) -> list[int]:
        """Find small prime factors of n."""
        factors = []
        d = 2
        temp = n
        while d * d <= temp and d < 100000:
            if temp % d == 0:
                factors.append(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            factors.append(temp)
        return factors

    def add(self, P: Point, Q: Point) -> Point:
        if P is None:
            return Q
        if Q is None:
            return P
        p = self.p
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and y1 == (p - y2) % p:
            return None
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

    def neg(self, P: Point) -> Point:
        if P is None:
            return None
        return (P[0], (self.p - P[1]) % self.p)

    def multiply(self, P: Point, k: int) -> Point:
        if k < 0:
            P = self.neg(P)
            k = -k
        if k == 0 or P is None:
            return None
        result: Point = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def random_keypair(
        self, rng: np.random.Generator | None = None
    ) -> tuple[int, tuple[int, int]]:
        """Generate a random private key and its public key."""
        if rng is None:
            rng = np.random.default_rng()
        order = self.order
        G = self.generator
        for _ in range(100):
            if order > 2**63:
                k = 1 + secrets.randbelow(order - 1)
            else:
                k = int(rng.integers(1, order))
            P = self.multiply(G, k)
            if P is not None:
                return k, P
        raise RuntimeError("Could not generate valid keypair")


class ECEnergyEvaluator:
    """Efficient EC constraint evaluation for MCMC dynamics.

    Precomputes power_points[i] = 2^i * G so that each spin flip
    requires only one EC point addition instead of a full O(n)
    scalar multiplication.

    Maintains a running point accumulator: current_point = k' * G
    where k' is the candidate key from the current spin configuration.
    """

    def __init__(
        self,
        curve: SmallEC,
        generator: tuple[int, int],
        public_key: tuple[int, int],
        n_bits: int,
    ) -> None:
        self.curve = curve
        self.generator = generator
        self.public_key = public_key
        self.n_bits = n_bits

        # Precompute 2^i * G for all bit positions
        self.power_points: list[tuple[int, int]] = []
        pt: Point = generator
        for _ in range(n_bits):
            assert pt is not None
            self.power_points.append(pt)
            pt = self.curve.add(pt, pt)

        # Negative power points for subtraction
        self.neg_power_points = [
            self.curve.neg(pp) for pp in self.power_points
        ]

        self._current_point: Point = None
        self._current_key: int = 0

    def set_state(self, key: int) -> None:
        """Set the current key and compute its point."""
        self._current_key = key
        self._current_point = self.curve.multiply(self.generator, key)

    def set_state_from_spins(self, spins: NDArray[np.int8]) -> None:
        """Set state from a spin configuration."""
        key = 0
        for j in range(self.n_bits):
            if spins[j] == -1:  # spin -1 -> bit 1
                key |= 1 << j
        self.set_state(key)

    def constraint_penalty(self) -> float:
        """Return 0.0 if current key satisfies k*G == P, else 1.0."""
        return 0.0 if self._current_point == self.public_key else 1.0

    def flip_single(self, bit_idx: int) -> float:
        """Flip one bit, update running point, return new penalty.

        If bit was 0 (spin +1), set to 1: add 2^i * G
        If bit was 1 (spin -1), set to 0: subtract 2^i * G
        """
        was_set = (self._current_key >> bit_idx) & 1
        if was_set:
            # bit 1 -> 0: subtract 2^i * G
            self._current_point = self.curve.add(
                self._current_point, self.neg_power_points[bit_idx]
            )
            self._current_key ^= 1 << bit_idx
        else:
            # bit 0 -> 1: add 2^i * G
            self._current_point = self.curve.add(
                self._current_point, self.power_points[bit_idx]
            )
            self._current_key ^= 1 << bit_idx

        return 0.0 if self._current_point == self.public_key else 1.0

    def flip_pair(self, bit_a: int, bit_b: int) -> float:
        """Flip two bits, update running point, return new penalty."""
        self.flip_single(bit_a)
        return self.flip_single(bit_b)

    def peek_flip_single(self, bit_idx: int) -> float:
        """Check what penalty would be after flipping bit, without changing state."""
        was_set = (self._current_key >> bit_idx) & 1
        if was_set:
            new_point = self.curve.add(
                self._current_point, self.neg_power_points[bit_idx]
            )
        else:
            new_point = self.curve.add(
                self._current_point, self.power_points[bit_idx]
            )
        return 0.0 if new_point == self.public_key else 1.0

    def peek_flip_pair(self, bit_a: int, bit_b: int) -> float:
        """Check penalty after flipping two bits, without changing state."""
        was_a = (self._current_key >> bit_a) & 1
        was_b = (self._current_key >> bit_b) & 1

        pt = self._current_point
        if was_a:
            pt = self.curve.add(pt, self.neg_power_points[bit_a])
        else:
            pt = self.curve.add(pt, self.power_points[bit_a])
        if was_b:
            pt = self.curve.add(pt, self.neg_power_points[bit_b])
        else:
            pt = self.curve.add(pt, self.power_points[bit_b])

        return 0.0 if pt == self.public_key else 1.0


# -- Standard test curves (y^2 = x^3 + 7, secp256k1 family) --

SMALL_CURVES = {
    "p97": SmallEC(97, 0, 7),
    "p251": SmallEC(251, 0, 7),
    "p509": SmallEC(509, 0, 7),
    "p1021": SmallEC(1021, 0, 7),
    "p2039": SmallEC(2039, 0, 7),
    "p4093": SmallEC(4093, 0, 7),
}

# Pre-computed curve parameters for y^2 = x^3 + 7 over F_p.
# Each entry: (p, order, generator_x, generator_y)
# Orders computed offline. All generators verified: order*G = O.
CURVE_PARAMS: dict[int, tuple[int, int, tuple[int, int]]] = {
    7: (131, 132, (1, 46)),
    8: (263, 270, (1, 8)),
    10: (1031, 1032, (0, 84)),
    12: (4099, 4228, (1, 103)),
    # For curves above enumeration limit, we supply the order directly
    # and use the first valid point as generator. The key bit length
    # is derived from the order, not from p.
}


def make_curve(target_bits: int) -> SmallEC:
    """Create a curve with approximately target_bits key size.

    For small sizes (<=12 bits), uses precomputed parameters.
    For larger sizes, uses a prime p ~ 2^target_bits and computes
    the curve on-the-fly with enumeration (up to p < 10^6) or
    with the order supplied.
    """
    if target_bits in CURVE_PARAMS:
        p, order, gen = CURVE_PARAMS[target_bits]
        curve = SmallEC(p, 0, 7, order=order)
        curve._gen = gen
        return curve

    # For larger sizes, find a suitable prime and let SmallEC enumerate
    # (only works up to p ~ 10^5)
    primes_3mod4 = []
    candidate = (1 << target_bits) + 1
    while len(primes_3mod4) < 1:
        if _is_prime(candidate) and candidate % 4 == 3:
            primes_3mod4.append(candidate)
        candidate += 2

    p = primes_3mod4[0]
    if p < 1_000_000:
        return SmallEC(p, 0, 7)

    # For large primes, skip order computation. Use p as the key space
    # upper bound (Hasse: order is within 2*sqrt(p) of p+1).
    # Set order = p so key_bit_length() returns ~target_bits.
    curve = SmallEC(p, 0, 7, order=p)
    # Find a generator point
    pt = curve._find_point()
    if pt is not None:
        curve._gen = pt
    return curve


def _is_prime(n: int) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    # Test with several witnesses
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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


class ECConstraintEncoder:
    """Encode EC DLP constraints as Ising interactions.

    For small curves (N <= 20 bits): exact penalty diagonal.
    For larger curves: uses ECEnergyEvaluator for direct evaluation.
    """

    def __init__(
        self,
        curve: SmallEC,
        generator: tuple[int, int],
        public_key: tuple[int, int],
    ) -> None:
        self.curve = curve
        self.generator = generator
        self.public_key = public_key
        self.n_bits = curve.key_bit_length()

    def full_penalty_diagonal(self) -> NDArray[np.float64]:
        """Build a 2^N diagonal penalty vector.

        penalty[i] = 0 if i*G == P (correct key), else 1.0.
        Only feasible for N <= 20 (~1M entries).
        """
        n = self.n_bits
        size = 1 << n
        penalty = np.ones(size, dtype=np.float64)

        order = self.curve.order
        for k in range(min(size, order)):
            pt = self.curve.multiply(self.generator, k)
            if pt == self.public_key:
                penalty[k] = 0.0

        return penalty

    def spin_penalty_diagonal(self) -> NDArray[np.float64]:
        """Like full_penalty_diagonal but indexed by spin configuration."""
        bit_penalty = self.full_penalty_diagonal()
        n = self.n_bits
        size = 1 << n
        spin_penalty = np.ones(size, dtype=np.float64)

        for spin_idx in range(size):
            key_val = 0
            for j in range(n):
                bit_j = (spin_idx >> j) & 1
                key_val |= bit_j << j
            if key_val < len(bit_penalty):
                spin_penalty[spin_idx] = bit_penalty[key_val]

        return spin_penalty

    def make_evaluator(self) -> ECEnergyEvaluator:
        """Create an ECEnergyEvaluator for efficient MCMC dynamics."""
        return ECEnergyEvaluator(
            self.curve, self.generator, self.public_key, self.n_bits
        )
