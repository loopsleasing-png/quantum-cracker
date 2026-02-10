#!/usr/bin/env python3
"""
Partial Key Exposure Attack on ECDLP / ECDSA
=============================================

Demonstrates how knowledge of partial bits of an elliptic curve private key
dramatically reduces the computational cost of recovering the full key.

This is directly relevant to real-world scenarios: cold boot attacks, side
channels, memory forensics, and fault injection can all leak partial key bits.

Requires: python3, ecdsa package (pip install ecdsa)
Run:  python scripts/experiment/partial_key_exposure.py
"""

from ecdsa.ellipticcurve import CurveFp, PointJacobi, INFINITY
import random
import time
import csv
import os
import math

# ---------------------------------------------------------------------------
# Output destination
# ---------------------------------------------------------------------------
DESKTOP = os.path.expanduser("~/Desktop")
CSV_PATH = os.path.join(DESKTOP, "partial_key_exposure.csv")
CSV_ROWS = []

SEPARATOR = "=" * 78
SUBSEP = "-" * 78


def record(curve_name, bits_total, bits_known, knowledge_type,
           search_space, time_ms, success):
    """Append one result row."""
    CSV_ROWS.append({
        "curve": curve_name, "bits_total": bits_total,
        "bits_known": bits_known, "knowledge_type": knowledge_type,
        "search_space": search_space, "time_ms": round(time_ms, 3),
        "success": success,
    })


# ---------------------------------------------------------------------------
# Small curve construction
# ---------------------------------------------------------------------------
def make_small_curve(p, a, b):
    """Build y^2 = x^3 + ax + b over F_p. Returns (curve, G, order) or None."""
    if (4 * a**3 + 27 * b**2) % p == 0:
        return None
    curve = CurveFp(p, a, b)
    for x in range(p):
        y_sq = (x**3 + a * x + b) % p
        if pow(y_sq, (p - 1) // 2, p) != 1:
            continue
        y = pow(y_sq, (p + 1) // 4, p)
        if (y * y) % p != y_sq:
            continue
        G = PointJacobi(curve, x, y, 1)
        P, order = G, 1
        for i in range(2, 2 * p + 10):
            P = P + G
            if P == INFINITY:
                order = i
                break
        if order > 20:
            return curve, G, order
    return None


SMALL_CURVES = [
    (251, 1, 1),   # order 282, ~9 bits
    (509, 1, 1),   # order 520, ~10 bits
    (1021, 1, 1),  # ~10-11 bits
    (251, 3, 2),   # order 132, ~8 bits
    (251, 4, 3),   # order 233, ~8 bits
]


def get_curves():
    """Return list of (name, curve, G, order) for experiments."""
    curves = []
    for p, a, b in SMALL_CURVES:
        result = make_small_curve(p, a, b)
        if result:
            curve, G, order = result
            curves.append((f"E(F_{p})[a={a},b={b}]", curve, G, order))
    return curves


# ---------------------------------------------------------------------------
# EC helpers
# ---------------------------------------------------------------------------
def scalar_mult(G, k, order=None):
    """Compute k * G on the curve."""
    if k == 0:
        return INFINITY
    if k < 0:
        P = G * (-k)
        if P == INFINITY:
            return INFINITY
        return PointJacobi(G.curve(), P.x(), (-P.y()) % G.curve().p(), 1)
    return G * k


def points_equal(P, Q):
    """Check if two EC points are equal."""
    if P == INFINITY and Q == INFINITY:
        return True
    if P == INFINITY or Q == INFINITY:
        return False
    return P.x() == Q.x() and P.y() == Q.y()


# ---------------------------------------------------------------------------
# PART 1: Baby-Step Giant-Step with Known Bits
# ---------------------------------------------------------------------------
def bsgs_with_known_bits(G, Q, order, known_bits_value, unknown_bit_count,
                         bit_position="lsb"):
    """
    BSGS search over the unknown portion of the key.

    bit_position == "lsb": d = known_bits_value * 2^unknown_bit_count + unknown
        (known bits are the MSBs, unknown are LSBs)
    """
    search_size = 1 << unknown_bit_count
    m = int(math.isqrt(search_size)) + 1

    # Baby steps: table of j*G for j in [0, m)
    baby = {}
    P = INFINITY
    for j in range(m):
        key = "inf" if P == INFINITY else (P.x(), P.y())
        baby[key] = j
        P = P + G

    neg_mG = scalar_mult(G, order - m, order)

    # Subtract known offset: Q - known_offset*G = unknown*G
    if bit_position == "lsb":
        offset = known_bits_value * (1 << unknown_bit_count)
    else:
        offset = known_bits_value

    neg_offset_G = scalar_mult(G, (order - offset) % order, order)
    offset_G = scalar_mult(G, offset % order, order)
    target = Q + neg_offset_G if offset_G != INFINITY else Q

    # Giant steps
    gamma = target
    for i in range(m):
        key = "inf" if gamma == INFINITY else (gamma.x(), gamma.y())
        if key in baby:
            j = baby[key]
            unknown = i * m + j
            if unknown < search_size:
                if bit_position == "lsb":
                    d_cand = (known_bits_value * (1 << unknown_bit_count) + unknown) % order
                else:
                    d_cand = (unknown * (1 << (order.bit_length() - unknown_bit_count)) + known_bits_value) % order
                if points_equal(scalar_mult(G, d_cand, order), Q):
                    return d_cand, i + 1
        gamma = gamma + neg_mG
    return None, m


def part1(curves):
    """BSGS with known bits: show speedup from partial key knowledge."""
    print(SEPARATOR)
    print("PART 1: Baby-Step Giant-Step with Known Bits")
    print(SEPARATOR)
    print()
    print("If the private key d has n bits and we know k of them (MSBs),")
    print("BSGS on the remaining (n-k) unknown bits costs O(2^((n-k)/2)).")
    print("We test with 0%, 25%, 50%, 75% of MSBs known.")
    print()

    for name, curve, G, order in curves:
        n = order.bit_length()
        if n < 7:
            continue

        print(SUBSEP)
        print(f"Curve: {name}    Order: {order}    Bits: {n}")
        print(SUBSEP)
        print(f"{'% Known':>10} {'Known':>8} {'Unknown':>9} "
              f"{'Search':>10} {'Steps':>8} {'ms':>8} {'Speedup':>9}")

        baseline_steps = None
        for pct in [0, 25, 50, 75]:
            bits_known = (n * pct) // 100
            bits_unknown = n - bits_known
            search_space = 1 << bits_unknown

            d = random.randint(1, order - 1)
            Q = scalar_mult(G, d, order)
            known_msb = d >> bits_unknown if bits_known > 0 else 0

            t0 = time.perf_counter()
            d_found, steps = bsgs_with_known_bits(
                G, Q, order, known_msb, bits_unknown, bit_position="lsb")
            elapsed_ms = (time.perf_counter() - t0) * 1000

            success = d_found is not None and d_found == d
            if baseline_steps is None and steps > 0:
                baseline_steps = steps
            speedup = (baseline_steps / steps) if baseline_steps and steps > 0 else 1.0

            status = "" if success else " [MISS]"
            print(f"{pct:>9}% {bits_known:>8} {bits_unknown:>9} "
                  f"{search_space:>10} {steps:>8} {elapsed_ms:>8.2f} "
                  f"{speedup:>8.1f}x{status}")

            record(name, n, bits_known, "msb", search_space, elapsed_ms, success)
        print()

    print("Takeaway: Each additional known bit roughly halves the BSGS step")
    print("count. Knowing 50% of bits reduces cost by ~2^(n/4) vs full search.")
    print()


# ---------------------------------------------------------------------------
# PART 2: Lattice-Based Recovery (MSB/LSB known)
# ---------------------------------------------------------------------------
def brute_force_with_known_msb(G, Q, order, d_high, bits_unknown):
    """d = d_high * 2^bits_unknown + x; search x in [0, 2^bits_unknown)."""
    shift = 1 << bits_unknown
    offset = (d_high * shift) % order
    target = Q + scalar_mult(G, (order - offset) % order, order)
    for x in range(min(shift, order)):
        if points_equal(scalar_mult(G, x, order), target):
            return (d_high * shift + x) % order, x + 1
    return None, min(shift, order)


def brute_force_with_known_lsb(G, Q, order, d_low, bits_known_low, n):
    """d = x * 2^bits_known_low + d_low; search x."""
    bits_unknown = n - bits_known_low
    shift = 1 << bits_known_low
    target = Q + scalar_mult(G, (order - d_low) % order, order)
    shiftG = scalar_mult(G, shift % order, order)
    P = INFINITY
    for x in range(min(1 << bits_unknown, order)):
        if points_equal(P, target):
            return (x * shift + d_low) % order, x + 1
        P = P + shiftG
    return None, min(1 << bits_unknown, order)


def part2(curves):
    """Lattice-based recovery: MSB and LSB knowledge."""
    print(SEPARATOR)
    print("PART 2: Lattice-Based Recovery (Coppersmith-Style)")
    print(SEPARATOR)
    print()
    print("Real lattice attacks (Coppersmith, Boneh-Durfee, Howgrave-Graham)")
    print("construct a short vector in a lattice when contiguous MSBs/LSBs known.")
    print("For toy curves we demonstrate with direct search:")
    print("  MSB known: d = d_high * 2^u + x, search x in [0, 2^u)")
    print("  LSB known: d = x * 2^k + d_low, search x in [0, 2^(n-k))")
    print()

    target_curves = [c for c in curves if c[3].bit_length() >= 8][:2]
    if not target_curves:
        target_curves = curves[:1]

    for name, curve, G, order in target_curves:
        n = order.bit_length()
        d = random.randint(1, order - 1)
        Q = scalar_mult(G, d, order)

        print(SUBSEP)
        print(f"Curve: {name}    Order: {order}    Bits: {n}")
        print(f"Secret key d = {d} = {bin(d)}")
        print(SUBSEP)

        for label, fracs, search_fn in [
            ("MSB Known", [0.25, 0.50, 0.625, 0.75, 0.875], "msb"),
            ("LSB Known", [0.25, 0.50, 0.625, 0.75, 0.875], "lsb"),
        ]:
            print(f"\n  [{label}]")
            print(f"  {'Known':>8} {'Unknown':>9} {'Search':>10} "
                  f"{'Steps':>8} {'ms':>8} {'OK':>5}")

            for frac in fracs:
                bits_known = int(n * frac)
                bits_unknown = n - bits_known
                if bits_known == 0 or bits_unknown == 0:
                    continue

                t0 = time.perf_counter()
                if search_fn == "msb":
                    d_found, steps = brute_force_with_known_msb(
                        G, Q, order, d >> bits_unknown, bits_unknown)
                else:
                    d_found, steps = brute_force_with_known_lsb(
                        G, Q, order, d & ((1 << bits_known) - 1), bits_known, n)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                success = d_found is not None and d_found == d

                print(f"  {bits_known:>8} {bits_unknown:>9} "
                      f"{1 << bits_unknown:>10} {steps:>8} "
                      f"{elapsed_ms:>8.2f} {'YES' if success else 'NO':>5}")
                record(name, n, bits_known, search_fn,
                       1 << bits_unknown, elapsed_ms, success)
        print()

    print("Takeaway: MSBs and LSBs give equal brute-force cost. But lattice")
    print("methods recover keys with as few as ~55% of MSBs on real curves,")
    print("exploiting ECDSA nonce algebraic structure beyond brute force.")
    print()


# ---------------------------------------------------------------------------
# PART 3: Random Bit Positions Known
# ---------------------------------------------------------------------------
def recover_with_random_bits(G, Q, order, d, known_positions, n):
    """Enumerate unknown bits, check each candidate against Q."""
    unknown_positions = [i for i in range(n) if i not in known_positions]
    num_unknown = len(unknown_positions)

    # Fixed part from known bits
    d_fixed = 0
    for pos in known_positions:
        d_fixed |= (((d >> pos) & 1) << pos)

    for combo in range(1 << num_unknown):
        d_candidate = d_fixed
        for idx, pos in enumerate(unknown_positions):
            if (combo >> idx) & 1:
                d_candidate |= (1 << pos)
        d_candidate %= order
        if d_candidate == 0:
            continue
        if points_equal(scalar_mult(G, d_candidate, order), Q):
            return d_candidate, combo + 1
    return None, 1 << num_unknown


def part3(curves):
    """Random bit positions known."""
    print(SEPARATOR)
    print("PART 3: Random Bit Positions Known")
    print(SEPARATOR)
    print()
    print("What if we know bits at *random* scattered positions (not contiguous)?")
    print("This models cold boot attacks where bit errors are uniformly distributed.")
    print()
    print("Search space is 2^(unknown bits) regardless of positions. But lattice")
    print("methods are LESS effective with random positions than contiguous MSBs.")
    print()

    small_curves = [c for c in curves if c[3].bit_length() <= 9][:2]
    if not small_curves:
        small_curves = curves[:1]

    for name, curve, G, order in small_curves:
        n = order.bit_length()
        d = random.randint(1, order - 1)
        Q = scalar_mult(G, d, order)

        print(SUBSEP)
        print(f"Curve: {name}    Order: {order}    Bits: {n}")
        print(f"Secret key d = {d} = {bin(d)}")
        print(SUBSEP)
        print(f"  {'%':>6} {'Known':>7} {'Unknown':>9} {'Search':>10} "
              f"{'AvgSteps':>10} {'ms':>8}")

        for pct in [0, 25, 50, 75, 90]:
            bits_known_count = max(0, (n * pct) // 100)
            bits_unknown_count = n - bits_known_count
            search_space = 1 << bits_unknown_count

            if search_space > 100000:
                print(f"  {pct:>5}% {bits_known_count:>7} {bits_unknown_count:>9} "
                      f"{search_space:>10} {'(skip)':>10} {'--':>8}")
                record(name, n, bits_known_count, "random", search_space, -1, False)
                continue

            total_steps, total_time, trials = 0, 0.0, 3
            all_success = True
            for _ in range(trials):
                d_t = random.randint(1, order - 1)
                Q_t = scalar_mult(G, d_t, order)
                positions = list(range(n))
                random.shuffle(positions)
                known_pos = set(positions[:bits_known_count])

                t0 = time.perf_counter()
                d_found, steps = recover_with_random_bits(G, Q_t, order, d_t, known_pos, n)
                total_time += (time.perf_counter() - t0) * 1000
                total_steps += steps
                if d_found != d_t:
                    all_success = False

            avg_s, avg_t = total_steps / trials, total_time / trials
            print(f"  {pct:>5}% {bits_known_count:>7} {bits_unknown_count:>9} "
                  f"{search_space:>10} {avg_s:>10.0f} {avg_t:>8.2f}")
            record(name, n, bits_known_count, "random", search_space, avg_t, all_success)
        print()

    print("Takeaway: Random-position bit knowledge gives 2^(n-k) search space,")
    print("same as contiguous. Lattice methods cannot exploit scattered positions.")
    print()


# ---------------------------------------------------------------------------
# PART 4: Scaling Analysis for secp256k1
# ---------------------------------------------------------------------------
def feasibility(cost_log2):
    if cost_log2 <= 20:
        return "EASY", "seconds"
    elif cost_log2 <= 40:
        return "FEASIBLE", "hours-days"
    elif cost_log2 <= 60:
        return "HARD", "years (cluster)"
    elif cost_log2 <= 80:
        return "EXTREME", "centuries"
    else:
        return "IMPOSSIBLE", "heat death"


def part4():
    """Extrapolate to 256-bit curves."""
    print(SEPARATOR)
    print("PART 4: Scaling Analysis for secp256k1 (256-bit)")
    print(SEPARATOR)
    print()
    print("Extrapolating from toy curves to production 256-bit keys.")
    print()
    print("Assumptions:")
    print("  - Classical BSGS: O(2^((n-k)/2)) group operations")
    print("  - Grover: O(2^((n-k)/2)) quantum ops (matches classical BSGS)")
    print("  - Lattice (Coppersmith): MSB knowledge, ~60% bits for ECDSA nonces")
    print("  - Feasibility: 2^40 ops = hours on 1 GPU; 2^20 = seconds")
    print()

    n = 256
    print(SUBSEP)
    print("SCALING TABLE: secp256k1 (n=256 bits)")
    print(SUBSEP)
    print(f"{'Known':>8} {'Unknown':>9} {'Method':>18} "
          f"{'Cost':>10} {'Feasible?':>12} {'Time':>16}")
    print(SUBSEP)

    for bits_known in [0, 64, 128, 192, 216, 226, 236, 240, 246, 250, 252, 254]:
        bits_unknown = n - bits_known
        classical = bits_unknown / 2.0
        grover = bits_unknown / 2.0
        lattice = max(0, bits_unknown * 0.8) if bits_known > 0 else bits_unknown

        # Classical BSGS
        f, t = feasibility(classical)
        print(f"{bits_known:>8} {bits_unknown:>9} {'Classical BSGS':>18} "
              f"{classical:>10.1f} {f:>12} {t:>16}")

        # Grover
        f_g, t_g = feasibility(grover)
        print(f"{'':>8} {'':>9} {'Grover':>18} "
              f"{grover:>10.1f} {f_g:>12} {t_g:>16}")

        # Lattice (MSB only)
        if bits_known > 0:
            f_l, t_l = feasibility(lattice)
            print(f"{'':>8} {'':>9} {'Lattice (MSB)':>18} "
                  f"{lattice:>10.1f} {f_l:>12} {t_l:>16}")
        print()

        record("secp256k1", 256, bits_known, "msb_theoretical",
               2**min(bits_unknown, 1023), -1, classical <= 40)

    print(SUBSEP)
    print()
    print("KEY THRESHOLDS FOR secp256k1:")
    print()
    print("  Classical BSGS feasible (2^40):   >= 176/256 bits known")
    print("  Classical BSGS comfortable (2^20): >= 216/256 bits known")
    print("  Grover matches classical BSGS for partial key (both sqrt)")
    print("  Grover + BSGS combined: potentially 2^((n-k)/3)")
    print()
    print("  Lattice (ECDSA nonce leakage):")
    print("    ~2-3 bits per nonce x 100 signatures = full key recovery")
    print("    (Howgrave-Graham & Smart, 2001; Boneh & Venkatesan, 1996)")
    print()
    print("  Bottom line: 87% known (223/256) = trivially feasible classical.")
    print()


# ---------------------------------------------------------------------------
# PART 5: Real-World Scenarios
# ---------------------------------------------------------------------------
def simulate_cold_boot(key_bits, error_rate):
    """Each bit has error_rate probability of corruption. Returns known positions."""
    known = [i for i in range(len(key_bits)) if random.random() > error_rate]
    return known


def part5():
    """Real-world attack scenarios."""
    print(SEPARATOR)
    print("PART 5: Real-World Partial Key Exposure Scenarios")
    print(SEPARATOR)
    print()
    print("Three realistic attack vectors on a 256-bit secp256k1 private key.")
    print()

    n = 256
    key_bits = [(random.getrandbits(256) >> (255 - i)) & 1 for i in range(256)]

    # ---- Scenario A: Cold Boot Attack ----
    print(SUBSEP)
    print("SCENARIO A: Cold Boot Attack")
    print(SUBSEP)
    print()
    print("DRAM retains data after power loss with increasing bit errors.")
    print("Halderman et al. (2008) measured error rates by cooling method:")
    print("  Liquid nitrogen (-196C): ~0.1% error after 60s")
    print("  Compressed air (-50C):   ~1% error after 30s")
    print("  Room temperature:        ~5-17% error after 5-15s")
    print()
    print(f"{'Cooling':>22} {'Error':>7} {'Correct':>9} "
          f"{'Wrong':>7} {'BSGS':>8} {'Feasible':>10}")
    print(SUBSEP)

    scenarios = [
        ("Liquid N2 (0.1%)", 0.001),
        ("Liquid N2 (0.5%)", 0.005),
        ("Compressed air (1%)", 0.01),
        ("Compressed air (3%)", 0.03),
        ("Room temp 5s (5%)", 0.05),
        ("Room temp 10s (10%)", 0.10),
        ("Room temp 15s (17%)", 0.17),
        ("No cooling 30s (35%)", 0.35),
    ]

    for scenario_name, error_rate in scenarios:
        avg_known = sum(len(simulate_cold_boot(key_bits, error_rate))
                        for _ in range(100)) / 100.0
        avg_unknown = n - avg_known
        cost = avg_unknown / 2.0
        f, _ = feasibility(cost)

        print(f"{scenario_name:>22} {error_rate:>6.1%} {avg_known:>9.1f} "
              f"{avg_unknown:>7.1f} {'2^%.0f' % cost:>8} {f:>10}")
        record("secp256k1", 256, int(avg_known), "random_coldboot",
               int(2 ** min(avg_unknown, 1023)), -1, cost <= 40)

    print()
    print("Even 1% error leaves ~2.6 unknown bits (BSGS 2^1.3 = trivial).")
    print("At 17% error: ~43 unknown bits (BSGS 2^22 = minutes).")
    print("Key only safe above ~35% error rate (BSGS 2^45 = hard).")
    print("Cold boot gives RANDOM errors -- lattice methods less helpful.")
    print()

    # ---- Scenario B: Side Channel (MSB Leakage) ----
    print(SUBSEP)
    print("SCENARIO B: Side-Channel MSB Leakage (DPA / Timing)")
    print(SUBSEP)
    print()
    print("DPA and timing attacks leak MSBs from scalar multiply's first ops.")
    print("ECDSA nonce leakage: 2-3 MSBs per sig x 100 sigs = full key (HNP).")
    print()
    print(f"{'MSBs':>8} {'Unknown':>9} {'BSGS':>10} "
          f"{'Lattice':>10} {'Assessment':>13}")
    print(SUBSEP)

    for msb in [4, 8, 16, 32, 64, 128, 192, 224, 240]:
        unk = n - msb
        bsgs = unk / 2.0
        latt = max(0, unk * 0.7)
        assess = ("TRIVIAL" if bsgs <= 20 else "FEASIBLE" if bsgs <= 40
                  else "HARD" if bsgs <= 80 else "INFEASIBLE")
        print(f"{msb:>8} {unk:>9} {'2^%.0f' % bsgs:>10} "
              f"{'2^%.0f' % latt:>10} {assess:>13}")
        record("secp256k1", 256, msb, "msb_sidechannel",
               2 ** min(unk, 1023), -1, bsgs <= 40)

    print()
    print("Contiguous MSBs are ideal for lattice recovery. Even 25% MSBs (64)")
    print("with lattice reduction puts the key at serious risk.")
    print()

    # ---- Scenario C: Memory Dump with Partial Overwrite ----
    print(SUBSEP)
    print("SCENARIO C: Memory Dump with Partial Overwrite")
    print(SUBSEP)
    print()
    print("Memory forensics recovers a 32-byte key with some bytes zeroed.")
    print("Models: heap spray after free(), partial secure erase, crash dump.")
    print()
    print(f"{'Intact':>8} {'Known':>8} {'Overwritten':>13} "
          f"{'BSGS':>8} {'Assessment':>12}")
    print(SUBSEP)

    for intact in [4, 8, 12, 16, 20, 24, 28, 30, 31]:
        over = 32 - intact
        bits_k = intact * 8
        bits_u = over * 8
        cost = bits_u / 2.0
        assess = ("TRIVIAL" if cost <= 20 else "FEASIBLE" if cost <= 40
                  else "HARD" if cost <= 60 else "EXTREME" if cost <= 80
                  else "IMPOSSIBLE")
        print(f"{intact:>6} B {bits_k:>8} {over:>11} B "
              f"{'2^%.0f' % cost:>8} {assess:>12}")
        record("secp256k1", 256, bits_k, "memory_contiguous",
               2 ** min(bits_u, 1023), -1, cost <= 40)

    print()
    print("28/32 bytes intact (87.5%) => BSGS 2^16 = trivial.")
    print("24/32 bytes intact (75%)   => BSGS 2^32 = hours.")
    print()

    # ---- Combined Summary ----
    print(SUBSEP)
    print("COMBINED SCENARIO SUMMARY")
    print(SUBSEP)
    print()
    print(f"{'Attack Vector':>32} {'Recovery':>10} "
          f"{'Unknown':>9} {'BSGS':>8} {'Risk':>12}")
    print(SUBSEP)

    summary = [
        ("Cold boot (LN2, 0.1% err)", "99.9%", 0.3, 0.1, "CRITICAL"),
        ("Cold boot (spray, 1% err)", "99.0%", 2.6, 1.3, "CRITICAL"),
        ("Cold boot (room, 10% err)", "90.0%", 25.6, 12.8, "HIGH"),
        ("Cold boot (no cool, 35%)", "65.0%", 89.6, 44.8, "MEDIUM"),
        ("DPA: 16 MSBs", "6.3%", 240, 120.0, "LOW (alone)"),
        ("DPA: 64 MSBs", "25.0%", 192, 96.0, "MEDIUM"),
        ("DPA: 128 MSBs", "50.0%", 128, 64.0, "HIGH"),
        ("ECDSA nonce: 2b x 100 sigs", "~100%", 0, 0, "CRITICAL"),
        ("Memory: 28/32 bytes", "87.5%", 32, 16.0, "CRITICAL"),
        ("Memory: 24/32 bytes", "75.0%", 64, 32.0, "HIGH"),
        ("Memory: 16/32 bytes", "50.0%", 128, 64.0, "MEDIUM"),
    ]

    for nm, rec, unk, cost, risk in summary:
        print(f"{nm:>32} {rec:>10} {unk:>9.0f} "
              f"{'2^%.0f' % cost:>8} {risk:>12}")

    print()
    print("DEFENSE RECOMMENDATIONS:")
    print("  1. Memory encryption (AMD SME/SEV, Intel TME/MKTME)")
    print("  2. Constant-time scalar multiplication (prevent side channels)")
    print("  3. RFC 6979 deterministic nonces (prevent nonce leakage)")
    print("  4. Key zeroization on process exit (prevent memory dumps)")
    print("  5. Hardware security modules (HSMs) for high-value keys")
    print()


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
def write_csv():
    """Write all collected results to CSV."""
    if not CSV_ROWS:
        return
    fieldnames = ["curve", "bits_total", "bits_known", "knowledge_type",
                  "search_space", "time_ms", "success"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in CSV_ROWS:
            r = dict(row)
            if isinstance(r["search_space"], (int, float)) and r["search_space"] > 10**300:
                r["search_space"] = "2^" + str(int(math.log2(r["search_space"])))
            writer.writerow(r)
    print(f"Results written to: {CSV_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print(SEPARATOR)
    print("  PARTIAL KEY EXPOSURE ATTACKS ON ECDLP / ECDSA")
    print("  Demonstrating how partial bit knowledge reduces key search space")
    print(SEPARATOR)
    print()
    print("Knowing a fraction of an EC private key's bits exponentially reduces")
    print("the cost of recovering the full key. This matters for:")
    print("  - Cold boot attacks (Halderman et al., 2008)")
    print("  - Side-channel attacks (DPA, timing, EM emanation)")
    print("  - Memory forensics and crash dump analysis")
    print("  - ECDSA nonce leakage (Howgrave-Graham & Smart, 2001)")
    print()
    print("Parts 1-3: toy EC curves (order ~100-1000)")
    print("Parts 4-5: extrapolation to secp256k1 (256-bit)")
    print()

    print("Building small test curves...")
    curves = get_curves()
    for name, _, G, order in curves:
        print(f"  {name}: G=({G.x()},{G.y()}), order={order}, "
              f"bits={order.bit_length()}")
    print()

    part1(curves)
    part2(curves)
    part3(curves)
    part4()
    part5()

    print(SEPARATOR)
    print("Writing CSV results...")
    print(SEPARATOR)
    write_csv()

    print()
    print(SEPARATOR)
    print("  EXPERIMENT COMPLETE")
    print(SEPARATOR)
    print()
    print("Key findings:")
    print()
    print("  1. BSGS with known bits: cost drops from O(2^(n/2)) to O(2^((n-k)/2)).")
    print("  2. MSB vs LSB vs Random: brute force cost depends only on count, not")
    print("     position. But lattice methods strongly favor contiguous MSBs.")
    print("  3. secp256k1: need ~176/256 bits for feasibility (2^40 BSGS).")
    print("  4. Cold boot with cooling: 99%+ recovery = trivial key extraction.")
    print("  5. ECDSA nonce leakage: 2-3 bits/sig x 100 sigs = full key via HNP.")
    print()


if __name__ == "__main__":
    main()
