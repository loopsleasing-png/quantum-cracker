"""Weak RNG and Brain Wallet Attack Demonstrations.

Shows how weak random number generation and deterministic key derivation
lead to Bitcoin private key theft. Six independent attack classes are
demonstrated, each corresponding to real historical incidents.

Historical context:
  - 2011-2015: Brain wallet sweeping bots stole hundreds of BTC
  - 2013: Android SecureRandom bug (insufficient PRNG entropy)
  - 2015: Blockchain.info weak RNG incident (0.0002 BTC entropy)
  - 2022: Profanity vanity address generator ($160M Wintermute hack)
  - Ongoing: Bitcoin "puzzle transactions" to sequential keys 1-256

Educational purpose only. Demonstrates why cryptographic key generation
MUST use a CSPRNG with full 256-bit entropy.

References:
  - Castellucci: "Cracking Cryptocurrency Brain Wallets" (DEF CON 2015)
  - Heninger et al: "Mining Your Ps and Qs" (USENIX Security 2012)
  - Courtois et al: "On the Insecurity of Brain Wallets" (2014)
  - CVE-2013-7440: Android SecureRandom PRNG seeding flaw
"""

from ecdsa import SECP256k1, SigningKey, VerifyingKey
from ecdsa.ellipticcurve import CurveFp, PointJacobi, INFINITY
import hashlib
import random
import time
import csv
import os
import struct
import secrets

# ====================================================================
# Constants
# ====================================================================

CURVE = SECP256k1
CURVE_ORDER = CURVE.order
GENERATOR = CURVE.generator
FIELD_P = CURVE.curve.p()

CSV_ROWS = []

# ====================================================================
# Utility Functions
# ====================================================================


def separator(char="=", width=78):
    """Print a separator line."""
    print(char * width)


def section_header(part_num, title):
    """Print a formatted section header."""
    print()
    separator()
    print(f"  PART {part_num}: {title}")
    separator()
    print()


def brain_wallet_key(passphrase):
    """Derive a Bitcoin private key from a passphrase via SHA-256.

    This is exactly how brain wallets work: the passphrase IS the key.
    Anyone who guesses the passphrase owns the funds.

    Args:
        passphrase: The human-readable passphrase string.

    Returns:
        (d, sk, vk): secret exponent, SigningKey, VerifyingKey
    """
    raw = hashlib.sha256(passphrase.encode("utf-8")).digest()
    d = int.from_bytes(raw, "big")
    d = d % CURVE_ORDER  # Reduce mod n to get valid scalar
    if d == 0:
        d = 1  # Edge case: zero is not a valid private key
    sk = SigningKey.from_secret_exponent(d, curve=SECP256k1)
    vk = sk.get_verifying_key()
    return d, sk, vk


def pubkey_to_address_hash(vk):
    """Compute a simplified address hash from a verifying key.

    Real Bitcoin uses RIPEMD160(SHA256(pubkey)) then Base58Check.
    We use truncated SHA256 for demonstration purposes.
    """
    pub_bytes = vk.to_string("compressed")
    h = hashlib.sha256(pub_bytes).hexdigest()[:40]
    return h


def int_to_private_key(n):
    """Convert an integer directly to a secp256k1 private key."""
    d = n % CURVE_ORDER
    if d == 0:
        d = 1
    sk = SigningKey.from_secret_exponent(d, curve=SECP256k1)
    vk = sk.get_verifying_key()
    return d, sk, vk


def ecdsa_sign_with_k(sk, msg_hash_int, k):
    """Sign a message hash using a specific nonce k.

    This bypasses normal nonce generation to demonstrate attacks.

    Args:
        sk: SigningKey object
        msg_hash_int: integer hash of the message
        k: the nonce to use (integer)

    Returns:
        (r, s): the signature components
    """
    d = sk.privkey.secret_multiplier
    n = CURVE_ORDER
    # R = k * G
    R = GENERATOR * k
    r = R.x() % n
    if r == 0:
        raise ValueError("r == 0, bad nonce")
    k_inv = pow(k, -1, n)
    s = (k_inv * (msg_hash_int + r * d)) % n
    if s == 0:
        raise ValueError("s == 0, bad nonce")
    return r, s


def recover_key_from_nonce_reuse(n, r, s1, h1, s2, h2):
    """Recover private key from two signatures sharing the same nonce.

    Math:
        s1 = k^{-1} * (h1 + r*d) mod n
        s2 = k^{-1} * (h2 + r*d) mod n

        s1 - s2 = k^{-1} * (h1 - h2) mod n
        k = (h1 - h2) * (s1 - s2)^{-1} mod n
        d = (s1*k - h1) * r^{-1} mod n

    Args:
        n: curve order
        r: shared r value (same k means same R point)
        s1, h1: first signature s-value and message hash
        s2, h2: second signature s-value and message hash

    Returns:
        Recovered private key d, or None on failure
    """
    ds = (s1 - s2) % n
    if ds == 0:
        return None
    dh = (h1 - h2) % n
    k = (dh * pow(ds, -1, n)) % n
    d = ((s1 * k - h1) * pow(r, -1, n)) % n
    return d


def format_hex(n, width=64):
    """Format an integer as zero-padded hex."""
    return f"0x{n:0{width}x}"


def format_time(seconds):
    """Format a duration for display."""
    if seconds < 1e-3:
        return f"{seconds*1e6:.1f} us"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    elif seconds < 365.25 * 86400:
        return f"{seconds/86400:.1f} days"
    else:
        return f"{seconds/(365.25*86400):.2e} years"


# ====================================================================
# Common password dictionaries
# ====================================================================

# Top 100 passwords used in brain wallet attacks (curated from
# real-world password lists + bitcoin-specific passphrases)
COMMON_PASSWORDS = [
    # Top passwords from various breach databases
    "password", "123456", "12345678", "qwerty", "abc123",
    "monkey", "1234567", "letmein", "trustno1", "dragon",
    "baseball", "iloveyou", "master", "sunshine", "ashley",
    "michael", "shadow", "123123", "654321", "superman",
    "qazwsx", "football", "password1", "password123", "batman",
    "login", "princess", "admin", "welcome", "hello",
    "charlie", "donald", "passw0rd", "whatever", "mustang",
    "access", "starwars", "1234", "12345", "123456789",
    "0", "1", "2", "3", "test",
    # Bitcoin / crypto specific passphrases
    "bitcoin", "satoshi", "nakamoto", "blockchain", "crypto",
    "ethereum", "litecoin", "dogecoin", "hodl", "tothemoon",
    "wallet", "mining", "hash", "block", "genesis",
    "satoshi nakamoto", "bitcoin is freedom", "in math we trust",
    # Famous passphrases
    "correct horse battery staple",  # XKCD 936
    "the quick brown fox",
    "to be or not to be",
    "i am satoshi nakamoto",
    "god", "love", "sex", "secret", "money",
    "freedom", "peace", "truth", "power", "magic",
    # Simple patterns
    "a", "b", "c", "aa", "aaa", "aaaa",
    "111111", "000000", "999999",
    "abcdef", "abcdefgh", "abcdefghij",
    "aaaaaaaa", "bbbbbbbb",
    "pass", "word", "passphrase", "brainwallet",
    "private", "key", "privatekey", "mykey", "mypassword",
    "asdfgh", "zxcvbn", "qwertyuiop",
    # Numbers as words
    "one", "two", "three", "four", "five",
    "zero", "null", "none", "empty", "blank",
]


# ====================================================================
# PART 1: Brain Wallet Attack
# ====================================================================

def part1_brain_wallet():
    section_header(1, "Brain Wallet Attack")

    print("  Brain wallets derive private keys via: SHA256(passphrase)")
    print("  Anyone who guesses the passphrase can steal all funds.")
    print("  Bots actively scan the blockchain for brain wallet deposits")
    print("  and sweep them within seconds of the first confirmation.")
    print()

    # --- Demo: specific well-known passphrases ---
    print("  [1a] Known passphrases -> instant key derivation")
    print("  " + "-" * 70)
    print()

    demo_phrases = [
        ("password", "Most common password globally"),
        ("bitcoin", "Obvious crypto passphrase"),
        ("satoshi", "Bitcoin creator's name"),
        ("correct horse battery staple", "XKCD 936 -- widely known"),
        ("1", "Single digit"),
        ("123456", "Sequential digits"),
        ("", "Empty string (yes, people did this)"),
        ("satoshi nakamoto", "Full pseudonym"),
    ]

    for phrase, note in demo_phrases:
        d, sk, vk = brain_wallet_key(phrase) if phrase else brain_wallet_key("")
        addr = pubkey_to_address_hash(vk)
        display_phrase = f'"{phrase}"' if phrase else '""'
        print(f"  Passphrase: {display_phrase:42s} [{note}]")
        print(f"    Private key: {format_hex(d)}")
        print(f"    Address:     {addr}")
        print()

    # --- Demo: dictionary scan timing ---
    print()
    print("  [1b] Dictionary attack: scanning 10,000 candidate passphrases")
    print("  " + "-" * 70)
    print()

    # Generate 10,000 candidate passphrases
    candidates = []
    # Include the 100 common passwords
    candidates.extend(COMMON_PASSWORDS)
    # Pad with numbered variants to reach 10,000
    for i in range(10000 - len(COMMON_PASSWORDS)):
        candidates.append(f"password{i}")

    t0 = time.time()
    derived_keys = {}
    for phrase in candidates:
        d, sk, vk = brain_wallet_key(phrase)
        derived_keys[phrase] = (d, pubkey_to_address_hash(vk))
    elapsed = time.time() - t0

    rate = len(candidates) / elapsed
    print(f"  Scanned {len(candidates):,} passphrases in {format_time(elapsed)}")
    print(f"  Rate: {rate:,.0f} keys/second (single CPU thread, pure Python)")
    print(f"  A GPU implementation achieves ~10^9 keys/sec per card.")
    print(f"  At that rate, a 1-billion-word dictionary takes ~1 second.")
    print()

    # --- Demo: full key recovery from dictionary ---
    print("  [1c] Full key recovery: 100 common passwords")
    print("  " + "-" * 70)
    print()

    # Simulate: a "victim" uses one of the common passwords
    # We pretend we know only the address, and we scan the dictionary
    victim_phrase = "correct horse battery staple"
    _, _, victim_vk = brain_wallet_key(victim_phrase)
    victim_addr = pubkey_to_address_hash(victim_vk)

    print(f"  Target address: {victim_addr}")
    print(f"  Scanning dictionary of {len(COMMON_PASSWORDS)} common passwords...")
    print()

    t0 = time.time()
    found = False
    keys_tested = 0
    for phrase in COMMON_PASSWORDS:
        keys_tested += 1
        d, sk, vk = brain_wallet_key(phrase)
        addr = pubkey_to_address_hash(vk)
        if addr == victim_addr:
            elapsed = time.time() - t0
            print(f"  RECOVERED after {keys_tested} attempts in {format_time(elapsed)}!")
            print(f"  Passphrase: \"{phrase}\"")
            print(f"  Private key: {format_hex(d)}")
            found = True
            break
    if not found:
        elapsed = time.time() - t0
        print(f"  Not found in dictionary ({keys_tested} tested, {format_time(elapsed)})")

    # Record CSV
    CSV_ROWS.append({
        "attack_type": "brain_wallet",
        "key_source": "SHA256(passphrase)",
        "entropy_bits": 20,  # ~1M common passwords = ~20 bits
        "keys_tested": keys_tested,
        "keys_found": 1 if found else 0,
        "time_ms": round(elapsed * 1000, 2),
        "effective_security": "20 bits (dictionary), 0 bits (common phrase)",
    })

    print()
    print("  Historical note: In 2013-2015, automated bots monitored the")
    print("  Bitcoin blockchain for brain wallet deposits. Funds sent to")
    print('  SHA256("password") were stolen within seconds. The address for')
    print('  SHA256("correct horse battery staple") received and lost over')
    print("  3 BTC across hundreds of transactions from people testing it.")
    print()

    return rate


# ====================================================================
# PART 2: Sequential / Low-Entropy Keys
# ====================================================================

def part2_sequential_keys():
    section_header(2, "Sequential / Low-Entropy Keys")

    print("  If private keys are generated from sequential integers or")
    print("  timestamps instead of 256-bit random values, the search")
    print("  space collapses from 2^256 to something trivially small.")
    print()

    # --- Demo: Bitcoin puzzle transactions ---
    print("  [2a] Bitcoin Puzzle Transactions")
    print("  " + "-" * 70)
    print()
    print("  In 2015, someone sent BTC to addresses derived from keys 1-256.")
    print("  These are real transactions on the Bitcoin blockchain. Keys 1-66")
    print("  have been claimed. Key 67+ remain (the search space grows).")
    print()

    print(f"  {'Key':>6s}   {'Private Key (hex)':>66s}   {'Address Hash':>42s}")
    print(f"  {'---':>6s}   {'---':>66s}   {'---':>42s}")

    puzzle_keys = [1, 2, 3, 4, 5, 10, 16, 32, 64, 128, 256]
    t0 = time.time()
    for key_int in puzzle_keys:
        d, sk, vk = int_to_private_key(key_int)
        addr = pubkey_to_address_hash(vk)
        print(f"  {key_int:6d}   {format_hex(d)}   {addr}")
    elapsed = time.time() - t0

    print()
    print(f"  Generated {len(puzzle_keys)} puzzle keys in {format_time(elapsed)}")
    print()

    # --- Demo: sequential scan ---
    print("  [2b] Sequential key scan: keys 1 through 1,000")
    print("  " + "-" * 70)
    print()

    t0 = time.time()
    sequential_keys = {}
    for i in range(1, 1001):
        d, sk, vk = int_to_private_key(i)
        sequential_keys[i] = pubkey_to_address_hash(vk)
    elapsed = time.time() - t0

    print(f"  Scanned keys 1-1000 in {format_time(elapsed)}")
    print(f"  Rate: {1000/elapsed:,.0f} keys/second")
    print()

    # --- Demo: timestamp-based keys ---
    print("  [2c] Timestamp-based keys: Unix timestamps as private keys")
    print("  " + "-" * 70)
    print()

    # Bitcoin genesis block: Jan 3, 2009 18:15:05 UTC
    genesis_ts = 1231006505
    # Current time (approximate)
    now_ts = int(time.time())
    total_seconds = now_ts - genesis_ts

    print(f"  Genesis block timestamp:  {genesis_ts}")
    print(f"  Current timestamp:        {now_ts}")
    print(f"  Total seconds in range:   {total_seconds:,} ({total_seconds.bit_length()} bits)")
    print()

    # Scan a sample window
    sample_window = 1000
    t0 = time.time()
    ts_keys = {}
    for ts in range(genesis_ts, genesis_ts + sample_window):
        d, sk, vk = int_to_private_key(ts)
        ts_keys[ts] = pubkey_to_address_hash(vk)
    elapsed = time.time() - t0

    print(f"  Scanned {sample_window} timestamps in {format_time(elapsed)}")
    rate = sample_window / elapsed
    full_scan_time = total_seconds / rate
    print(f"  Rate: {rate:,.0f} keys/second (Python)")
    print(f"  Full scan of all timestamps: {format_time(full_scan_time)} (Python)")
    print(f"  At 10^9 keys/sec (GPU): {format_time(total_seconds / 1e9)}")
    print()

    # --- Entropy comparison ---
    print("  [2d] Key space reduction analysis")
    print("  " + "-" * 70)
    print()

    comparisons = [
        ("Proper 256-bit random", 256, 2**256),
        ("Sequential integer (1-2^32)", 32, 2**32),
        ("Unix timestamp (2009-2026)", 31, total_seconds),
        ("Millisecond timestamp", 41, total_seconds * 1000),
        ("Sequential + 16-bit random", 48, 2**32 * 2**16),
        ("Brain wallet (1M dictionary)", 20, 10**6),
        ("Brain wallet (1B dictionary)", 30, 10**9),
        ("4-word passphrase (2048 words)", 44, 2048**4),
    ]

    print(f"  {'Key Source':<35s} {'Entropy':>10s} {'Key Space':>15s} {'GPU Scan Time':>18s}")
    print(f"  {'-'*35:<35s} {'-'*10:>10s} {'-'*15:>15s} {'-'*18:>18s}")
    for label, bits, space in comparisons:
        gpu_time = space / 1e9  # seconds at 10^9 keys/sec
        print(f"  {label:<35s} {bits:>7d} bit {'~2^'+str(bits):>15s} {format_time(gpu_time):>18s}")

    print()
    print("  A 256-bit random key requires 2^256 / 10^9 = 3.67 * 10^67 seconds")
    print("  to brute-force at 10^9 keys/sec. That is ~1.16 * 10^60 years,")
    print("  far longer than the age of the universe (1.38 * 10^10 years).")
    print()

    CSV_ROWS.append({
        "attack_type": "sequential_key",
        "key_source": "integers 1-1000",
        "entropy_bits": 10,
        "keys_tested": 1000,
        "keys_found": 1000,
        "time_ms": round(elapsed * 1000, 2),
        "effective_security": "10 bits (known pattern)",
    })

    CSV_ROWS.append({
        "attack_type": "timestamp_key",
        "key_source": "Unix timestamps 2009-2026",
        "entropy_bits": 31,
        "keys_tested": sample_window,
        "keys_found": sample_window,
        "time_ms": round(elapsed * 1000, 2),
        "effective_security": "31 bits (timestamp range)",
    })


# ====================================================================
# PART 3: Android SecureRandom Bug (2013)
# ====================================================================

def part3_android_securerandom():
    section_header(3, "Android SecureRandom Bug (CVE-2013-7440)")

    print("  In August 2013, a critical vulnerability was discovered in")
    print("  Android's Java SecureRandom implementation. The PRNG was")
    print("  initialized with insufficient entropy, causing multiple")
    print("  Bitcoin wallet apps to generate the same ECDSA nonce k")
    print("  for different transactions.")
    print()
    print("  When two ECDSA signatures share the same nonce k:")
    print("    s1 = k^{-1} * (h1 + r*d) mod n")
    print("    s2 = k^{-1} * (h2 + r*d) mod n")
    print("  Subtracting: k = (h1-h2) / (s1-s2) mod n")
    print("  Then:        d = (s1*k - h1) / r mod n")
    print()
    print("  The private key d is recovered from public blockchain data.")
    print()

    # --- Demo: nonce reuse attack on secp256k1 ---
    print("  [3a] Nonce reuse demonstration on secp256k1")
    print("  " + "-" * 70)
    print()

    # Generate a victim key
    victim_d = secrets.randbelow(CURVE_ORDER - 1) + 1
    victim_sk = SigningKey.from_secret_exponent(victim_d, curve=SECP256k1)

    # The bug: same nonce k used for two different messages
    bad_k = secrets.randbelow(CURVE_ORDER - 1) + 1

    # Two different "transaction hashes"
    h1 = int.from_bytes(hashlib.sha256(b"transaction_1_data").digest(), "big") % CURVE_ORDER
    h2 = int.from_bytes(hashlib.sha256(b"transaction_2_data").digest(), "big") % CURVE_ORDER

    # Sign both with the SAME nonce
    r1, s1 = ecdsa_sign_with_k(victim_sk, h1, bad_k)
    r2, s2 = ecdsa_sign_with_k(victim_sk, h2, bad_k)

    print(f"  Victim private key:  {format_hex(victim_d)}")
    print(f"  Reused nonce k:      {format_hex(bad_k)}")
    print()
    print(f"  Signature 1: r = {format_hex(r1)}")
    print(f"               s = {format_hex(s1)}")
    print(f"               h = {format_hex(h1)}")
    print()
    print(f"  Signature 2: r = {format_hex(r2)}")
    print(f"               s = {format_hex(s2)}")
    print(f"               h = {format_hex(h2)}")
    print()

    assert r1 == r2, "Same k must produce same r"
    print(f"  r1 == r2: True (same k -> same R point -> same r)")
    print()

    # Attack: recover private key
    t0 = time.time()
    recovered_d = recover_key_from_nonce_reuse(CURVE_ORDER, r1, s1, h1, s2, h2)
    elapsed = time.time() - t0

    if recovered_d == victim_d:
        print(f"  PRIVATE KEY RECOVERED in {format_time(elapsed)}!")
        print(f"  Recovered: {format_hex(recovered_d)}")
        print(f"  Matches:   True")
    else:
        print(f"  Recovery failed (recovered={recovered_d}, expected={victim_d})")

    print()

    CSV_ROWS.append({
        "attack_type": "android_nonce_reuse",
        "key_source": "SecureRandom(insufficient entropy)",
        "entropy_bits": 0,
        "keys_tested": 1,
        "keys_found": 1 if recovered_d == victim_d else 0,
        "time_ms": round(elapsed * 1000, 4),
        "effective_security": "0 bits (algebraic recovery)",
    })

    # --- Demo: batch detection ---
    print("  [3b] Detecting nonce reuse in a batch of signatures")
    print("  " + "-" * 70)
    print()

    # Simulate 20 signatures, some with reused nonces
    print("  Simulating 20 ECDSA signatures from the same key...")
    print("  Deliberately reusing nonce for signatures #5 and #12.")
    print()

    sigs = []
    reused_k = secrets.randbelow(CURVE_ORDER - 1) + 1
    for i in range(20):
        msg = f"transaction_{i}_data".encode()
        h = int.from_bytes(hashlib.sha256(msg).digest(), "big") % CURVE_ORDER
        if i in (5, 12):
            k = reused_k  # Bug: same nonce
        else:
            k = secrets.randbelow(CURVE_ORDER - 1) + 1
        r, s = ecdsa_sign_with_k(victim_sk, h, k)
        sigs.append((i, r, s, h))

    # Scan for repeated r values
    r_map = {}
    for idx, r, s, h in sigs:
        if r not in r_map:
            r_map[r] = []
        r_map[r].append((idx, s, h))

    collisions = {r: entries for r, entries in r_map.items() if len(entries) > 1}
    print(f"  Scanned 20 signatures for repeated r values.")
    print(f"  Found {len(collisions)} r-value collision(s).")

    for r, entries in collisions.items():
        print(f"  Collision: r = {format_hex(r)}")
        for idx, s, h in entries:
            print(f"    sig #{idx}: s={format_hex(s)[:20]}... h={format_hex(h)[:20]}...")
        # Recover key from the first pair
        idx1, s1, h1 = entries[0]
        idx2, s2, h2 = entries[1]
        d_recov = recover_key_from_nonce_reuse(CURVE_ORDER, r, s1, h1, s2, h2)
        if d_recov == victim_d:
            print(f"    -> Key recovered from sigs #{idx1} and #{idx2}")
    print()

    print("  Impact: An estimated 55 BTC was stolen from Android wallets")
    print("  before Google patched SecureRandom in August 2013. The fix")
    print("  added /dev/urandom seeding to the PRNG initialization.")
    print()


# ====================================================================
# PART 4: Profanity Vanity Address Generator (2022)
# ====================================================================

def part4_profanity_vanity():
    section_header(4, "Profanity Vanity Address Generator (Wintermute, 2022)")

    print("  Profanity was a popular Ethereum vanity address generator.")
    print("  It generated keys by: private_key = seed + increment")
    print("  where seed was derived from a GPU-computed random value")
    print("  with only 32 bits of entropy.")
    print()
    print("  Attack: knowing ANY one generated key reveals the seed,")
    print("  which compromises ALL keys generated from that seed.")
    print()
    print("  In September 2022, attackers exploited this to steal")
    print("  $160 million from Wintermute's hot wallet.")
    print()

    # --- Demo: seed + increment pattern ---
    print("  [4a] Demonstrating the seed + increment vulnerability")
    print("  " + "-" * 70)
    print()

    # Simulate Profanity's key generation with a weak seed
    # Real Profanity used 32-bit seeds; we demonstrate the pattern
    weak_seed = secrets.randbelow(2**32)  # Only 32 bits of entropy
    num_generated = 20

    print(f"  Weak seed (32-bit): {weak_seed} (0x{weak_seed:08x})")
    print(f"  Generating {num_generated} keys via seed + increment:")
    print()

    generated_keys = []
    for i in range(num_generated):
        d = (weak_seed + i) % CURVE_ORDER
        if d == 0:
            d = 1
        sk = SigningKey.from_secret_exponent(d, curve=SECP256k1)
        vk = sk.get_verifying_key()
        addr = pubkey_to_address_hash(vk)
        generated_keys.append((i, d, addr))
        if i < 5 or i == num_generated - 1:
            print(f"  key[{i:2d}]: d = ...{format_hex(d)[-16:]}  addr = {addr[:16]}...")
        elif i == 5:
            print(f"  {'...':>8s}")

    print()

    # --- Demo: recovering seed from one known key ---
    print("  [4b] Recovering seed from one compromised key")
    print("  " + "-" * 70)
    print()

    # Attacker knows key[7] was compromised (e.g., used on-chain)
    compromised_idx = 7
    known_d = generated_keys[compromised_idx][1]
    print(f"  Attacker knows private key at index {compromised_idx}:")
    print(f"    d = {format_hex(known_d)}")
    print()

    # Recover the seed: seed = d - index
    recovered_seed = (known_d - compromised_idx) % CURVE_ORDER
    # Since seed was 32-bit, reduce
    recovered_seed_32 = recovered_seed % (2**32)

    print(f"  Recovered seed: {recovered_seed_32} (0x{recovered_seed_32:08x})")
    print(f"  Original seed:  {weak_seed} (0x{weak_seed:08x})")
    print(f"  Match: {recovered_seed_32 == weak_seed}")
    print()

    # Now recover ALL other keys
    print("  Recovering all keys from the seed:")
    all_recovered = True
    for orig_i, orig_d, orig_addr in generated_keys:
        test_d = (recovered_seed_32 + orig_i) % CURVE_ORDER
        if test_d != orig_d:
            all_recovered = False
            print(f"    key[{orig_i}]: MISMATCH")
        elif orig_i < 5 or orig_i == num_generated - 1:
            print(f"    key[{orig_i:2d}]: recovered correctly  addr = {orig_addr[:16]}...")
        elif orig_i == 5:
            print(f"    {'...':>8s}")

    print()
    if all_recovered:
        print(f"  All {num_generated} keys recovered from a single compromised key.")
    print()

    # --- Demo: brute force search space ---
    print("  [4c] Brute-force search over 32-bit seed space")
    print("  " + "-" * 70)
    print()

    # Show how fast we can search 32-bit seeds
    target_addr = generated_keys[0][2]  # Address from key[0] = seed itself
    sample_size = 10000

    t0 = time.time()
    found_seed = None
    for candidate in range(sample_size):
        d = candidate % CURVE_ORDER
        if d == 0:
            d = 1
        sk = SigningKey.from_secret_exponent(d, curve=SECP256k1)
        vk = sk.get_verifying_key()
        addr = pubkey_to_address_hash(vk)
        if addr == target_addr:
            found_seed = candidate
            break
    elapsed = time.time() - t0

    rate = sample_size / elapsed
    full_32bit_time = 2**32 / rate

    print(f"  Scanned {sample_size:,} seed candidates in {format_time(elapsed)}")
    print(f"  Rate: {rate:,.0f} seeds/second (Python)")
    print(f"  Full 2^32 search: {format_time(full_32bit_time)} (Python)")
    print(f"  At 10^9 seeds/sec (GPU): {format_time(2**32 / 1e9)}")
    print(f"  At 10^12 seeds/sec (cluster): {format_time(2**32 / 1e12)}")
    print()

    CSV_ROWS.append({
        "attack_type": "profanity_vanity",
        "key_source": "seed(32-bit) + increment",
        "entropy_bits": 32,
        "keys_tested": sample_size,
        "keys_found": 1 if found_seed is not None else 0,
        "time_ms": round(elapsed * 1000, 2),
        "effective_security": "32 bits (seed entropy)",
    })

    print("  Timeline of the Wintermute hack:")
    print("  - 2022-06: 1inch team publicly discloses Profanity vulnerability")
    print("  - 2022-09-15: Amber Group demonstrates key recovery technique")
    print("  - 2022-09-20: Attacker drains $160M from Wintermute hot wallet")
    print("  - The hot wallet address was generated by Profanity")
    print("  - Attacker brute-forced the 32-bit seed space to recover the key")
    print()


# ====================================================================
# PART 5: Repeated r-Value Detection
# ====================================================================

def part5_repeated_r_detection():
    section_header(5, "Repeated r-Value Detection (Blockchain Forensics)")

    print("  Every ECDSA signature (r, s) is publicly visible on the")
    print("  blockchain. If any two signatures from the same key share")
    print("  the same r value, the nonce was reused and the private key")
    print("  is immediately recoverable.")
    print()
    print("  This is a passive attack: no interaction with the victim needed.")
    print("  Just scan the blockchain for r-value collisions.")
    print()

    # --- Demo: large-scale simulation ---
    print("  [5a] Generating 1000 signatures with deliberate nonce reuse")
    print("  " + "-" * 70)
    print()

    # Generate a victim key
    victim_d = secrets.randbelow(CURVE_ORDER - 1) + 1
    victim_sk = SigningKey.from_secret_exponent(victim_d, curve=SECP256k1)

    # Generate 1000 signatures, with 5 pairs of reused nonces
    reuse_pairs = [(42, 317), (88, 555), (201, 743), (333, 901), (450, 678)]
    reuse_nonces = {}
    for a, b in reuse_pairs:
        k = secrets.randbelow(CURVE_ORDER - 1) + 1
        reuse_nonces[a] = k
        reuse_nonces[b] = k

    all_sigs = []
    t0_gen = time.time()
    for i in range(1000):
        msg = f"tx_{i:04d}_{secrets.token_hex(8)}".encode()
        h = int.from_bytes(hashlib.sha256(msg).digest(), "big") % CURVE_ORDER
        if i in reuse_nonces:
            k = reuse_nonces[i]
        else:
            k = secrets.randbelow(CURVE_ORDER - 1) + 1
        r, s = ecdsa_sign_with_k(victim_sk, h, k)
        all_sigs.append((i, r, s, h))
    gen_time = time.time() - t0_gen

    print(f"  Generated 1000 signatures in {format_time(gen_time)}")
    print(f"  Planted 5 nonce-reuse pairs: {reuse_pairs}")
    print()

    # --- Detection: scan for repeated r ---
    print("  [5b] Detection: scanning for r-value collisions")
    print("  " + "-" * 70)
    print()

    t0_scan = time.time()
    r_index = {}  # r -> list of (sig_index, s, h)
    for idx, r, s, h in all_sigs:
        if r not in r_index:
            r_index[r] = []
        r_index[r].append((idx, s, h))
    scan_time = time.time() - t0_scan

    collisions = {r: entries for r, entries in r_index.items() if len(entries) > 1}

    print(f"  Scanned {len(all_sigs)} signatures in {format_time(scan_time)}")
    print(f"  Unique r values: {len(r_index)}")
    print(f"  r-value collisions found: {len(collisions)}")
    print()

    # --- Exploitation: recover keys from collisions ---
    print("  [5c] Exploitation: recovering private key from each collision")
    print("  " + "-" * 70)
    print()

    keys_recovered = 0
    for r, entries in sorted(collisions.items(), key=lambda x: x[1][0][0]):
        idx1, s1, h1 = entries[0]
        idx2, s2, h2 = entries[1]

        t0_attack = time.time()
        recovered = recover_key_from_nonce_reuse(CURVE_ORDER, r, s1, h1, s2, h2)
        attack_time = time.time() - t0_attack

        success = (recovered == victim_d)
        if success:
            keys_recovered += 1

        print(f"  Collision: sigs #{idx1} and #{idx2}")
        print(f"    r = {format_hex(r)[:32]}...")
        print(f"    Key recovered: {success}  ({format_time(attack_time)})")

    print()
    print(f"  Result: {keys_recovered}/{len(collisions)} collisions yielded the private key")
    print()

    CSV_ROWS.append({
        "attack_type": "r_value_reuse",
        "key_source": "ECDSA nonce reuse detection",
        "entropy_bits": 0,
        "keys_tested": len(all_sigs),
        "keys_found": keys_recovered,
        "time_ms": round((scan_time + gen_time) * 1000, 2),
        "effective_security": "0 bits (algebraic from r-collision)",
    })

    # --- Scaling analysis ---
    print("  [5d] Scaling to blockchain-sized data")
    print("  " + "-" * 70)
    print()

    # Bitcoin has ~900M signatures as of 2024
    blockchain_sigs = 900_000_000
    # Hash map lookup is O(1), insertion is O(1) amortized
    # Memory: ~64 bytes per entry (r_value + metadata)
    memory_gb = (blockchain_sigs * 64) / (1024**3)

    print(f"  Bitcoin blockchain: ~{blockchain_sigs/1e6:.0f}M ECDSA signatures (as of 2024)")
    print(f"  Hash map memory: ~{memory_gb:.1f} GB (r-value index)")
    print(f"  Scan time: O(n) = linear, single pass over all signatures")
    print(f"  Detection: O(1) per signature (hash map lookup)")
    print()
    print("  Known results from blockchain analysis:")
    print("  - Researchers have found hundreds of nonce-reuse incidents")
    print("  - Most from the 2011-2013 era (poor wallet implementations)")
    print("  - Some from the 2013 Android SecureRandom bug")
    print("  - Automated bots continuously scan for new reuses")
    print()


# ====================================================================
# PART 6: Entropy Analysis and Scaling
# ====================================================================

def part6_entropy_analysis():
    section_header(6, "Entropy Analysis and Scaling")

    print("  Comprehensive comparison of effective security across all")
    print("  attack classes, with realistic timing estimates for different")
    print("  attacker capabilities.")
    print()

    # --- Attack class comparison ---
    print("  [6a] Attack class comparison table")
    print("  " + "-" * 70)
    print()

    # Define attack classes with their parameters
    attack_classes = [
        {
            "name": "Proper 256-bit key",
            "entropy_bits": 256,
            "description": "CSPRNG with full entropy",
            "real_world": "Bitcoin Core default",
        },
        {
            "name": "Brain wallet (top 1M)",
            "entropy_bits": 20,
            "description": "SHA256(common passphrase)",
            "real_world": "2011-2015 brain wallet theft",
        },
        {
            "name": "Brain wallet (top 1B)",
            "entropy_bits": 30,
            "description": "SHA256(dictionary word)",
            "real_world": "Automated dictionary attack",
        },
        {
            "name": "Sequential integer",
            "entropy_bits": 32,
            "description": "key = counter (32-bit)",
            "real_world": "Puzzle transactions",
        },
        {
            "name": "Timestamp (seconds)",
            "entropy_bits": 31,
            "description": "key = Unix timestamp",
            "real_world": "Weak time-based seeding",
        },
        {
            "name": "Timestamp (ms)",
            "entropy_bits": 41,
            "description": "key = ms timestamp",
            "real_world": "Slightly better time seeding",
        },
        {
            "name": "Profanity (32-bit seed)",
            "entropy_bits": 32,
            "description": "seed(32) + increment",
            "real_world": "Wintermute $160M hack (2022)",
        },
        {
            "name": "Android SecureRandom",
            "entropy_bits": 0,
            "description": "Nonce reuse (same k)",
            "real_world": "CVE-2013-7440, 55 BTC stolen",
        },
        {
            "name": "ECDSA nonce reuse",
            "entropy_bits": 0,
            "description": "Two sigs, same k",
            "real_world": "Sony PS3 (2010), many wallets",
        },
        {
            "name": "Biased nonce (1 bit)",
            "entropy_bits": 255,
            "description": "Top bit of k always 0",
            "real_world": "Minerva (2019), LadderLeak (2020)",
        },
        {
            "name": "Java Random (48-bit)",
            "entropy_bits": 48,
            "description": "java.util.Random as key",
            "real_world": "Early Android apps",
        },
        {
            "name": "PHP mt_rand (32-bit)",
            "entropy_bits": 32,
            "description": "Mersenne Twister seed",
            "real_world": "PHP web wallets",
        },
    ]

    # GPU cluster: 10^9 keys/sec
    # Nation-state: 10^12 keys/sec
    gpu_rate = 1e9
    nation_rate = 1e12

    header = (f"  {'Attack Class':<28s} {'Entropy':>8s} {'Key Space':>14s} "
              f"{'GPU (10^9/s)':>16s} {'Nation (10^12)':>16s}")
    print(header)
    print(f"  {'-'*28} {'-'*8} {'-'*14} {'-'*16} {'-'*16}")

    for ac in attack_classes:
        bits = ac["entropy_bits"]
        if bits == 0:
            space_str = "1 (algebra)"
            gpu_str = "instant"
            nation_str = "instant"
        elif bits <= 64:
            space = 2 ** bits
            space_str = f"2^{bits}"
            gpu_str = format_time(space / gpu_rate)
            nation_str = format_time(space / nation_rate)
        elif bits == 255:
            space_str = f"2^{bits}"
            # Lattice attack, not brute force -- needs ~200 sigs
            gpu_str = "~200 sigs *"
            nation_str = "~200 sigs *"
        else:
            space_str = f"2^{bits}"
            gpu_str = "infeasible"
            nation_str = "infeasible"

        print(f"  {ac['name']:<28s} {bits:>5d} bit {space_str:>14s} "
              f"{gpu_str:>16s} {nation_str:>16s}")

    print()
    print("  * Biased nonce attacks use lattice reduction (HNP), not brute force.")
    print("    They need ~200 signatures, not 2^255 operations.")
    print()

    # --- Historical timeline ---
    print("  [6b] Historical timeline of real Bitcoin thefts from weak RNG")
    print("  " + "-" * 70)
    print()

    timeline = [
        ("2011-03", "Brain wallet sweepers appear",
         "Bots scan for SHA256(common_phrase) deposits",
         "Ongoing, hundreds of BTC"),
        ("2012-08", "Blockchain.info RNG flaw",
         "Insufficient entropy in key generation",
         "Affected subset of wallets"),
        ("2013-08", "Android SecureRandom bug",
         "PRNG seeded with insufficient entropy",
         "~55 BTC stolen, CVE-2013-7440"),
        ("2014-01", "ECDSA nonce reuse on blockchain",
         "Researchers find reused nonces in old txs",
         "Multiple keys recovered"),
        ("2015-01", "Bitcoin puzzle transactions",
         "Someone funds keys 1-256 as a challenge",
         "Keys 1-66 claimed, rest pending"),
        ("2015-08", "DEF CON brain wallet talk",
         "Castellucci presents 'Cracking Brain Wallets'",
         "All simple brain wallets swept"),
        ("2017-11", "Parity multi-sig library freeze",
         "Not RNG but related: predictable addresses",
         "$150M locked permanently"),
        ("2019-10", "Minerva attack published",
         "Timing-based nonce bias in smart cards",
         "CVE-2019-15809"),
        ("2020-09", "LadderLeak published",
         "1-bit nonce bias from non-constant-time code",
         "Academic demonstration"),
        ("2022-01", "Profanity vulnerability disclosed",
         "32-bit seed entropy in vanity generator",
         "1inch team publishes warning"),
        ("2022-09", "Wintermute hack",
         "Attacker exploits Profanity-generated address",
         "$160M stolen"),
        ("2023-11", "Randstorm disclosure",
         "BitcoinJS (2011-2015) had weak RNG",
         "Millions of wallets affected"),
    ]

    print(f"  {'Date':<10s} {'Event':<35s} {'Mechanism':<42s}")
    print(f"  {'-'*10} {'-'*35} {'-'*42}")
    for date, event, mechanism, impact in timeline:
        print(f"  {date:<10s} {event:<35s} {mechanism:<42s}")
        print(f"  {'':>10s} Impact: {impact}")

    print()

    # --- CSV entries for all attack classes ---
    for ac in attack_classes:
        bits = ac["entropy_bits"]
        if bits == 0:
            gpu_time = 0
        elif bits <= 64:
            gpu_time = (2**bits / gpu_rate) * 1000  # ms
        else:
            gpu_time = float("inf")

        CSV_ROWS.append({
            "attack_type": ac["name"].lower().replace(" ", "_").replace("(", "").replace(")", ""),
            "key_source": ac["description"],
            "entropy_bits": bits,
            "keys_tested": "N/A",
            "keys_found": "N/A",
            "time_ms": f"{gpu_time:.2f}" if gpu_time != float("inf") else "infeasible",
            "effective_security": f"{bits} bits" + (f" ({ac['real_world']})" if ac.get("real_world") else ""),
        })

    # --- Defense summary ---
    print("  [6c] Defense requirements")
    print("  " + "-" * 70)
    print()

    print("  For key generation:")
    print("    1. Use a CSPRNG (os.urandom, /dev/urandom, CryptGenRandom)")
    print("    2. Generate FULL 256-bit random keys (not derived from passwords)")
    print("    3. NEVER use brain wallets, sequential numbers, or timestamps")
    print("    4. Verify entropy source at runtime (check /dev/random available)")
    print()
    print("  For ECDSA signing:")
    print("    1. Use RFC 6979 deterministic nonce generation")
    print("    2. Use constant-time scalar multiplication")
    print("    3. Validate all intermediate values (r != 0, s != 0)")
    print("    4. Consider Schnorr signatures (BIP 340) for simpler security")
    print()
    print("  For wallet software:")
    print("    1. Use well-audited libraries (libsecp256k1, not custom code)")
    print("    2. BIP 32/39/44 hierarchical deterministic wallets")
    print("    3. Hardware wallets for significant holdings")
    print("    4. Multi-signature setups for organizational funds")
    print()


# ====================================================================
# Main
# ====================================================================

def main():
    print()
    separator("=")
    print("  WEAK RNG AND BRAIN WALLET ATTACK DEMONSTRATIONS")
    print("  How predictable key generation leads to Bitcoin theft")
    separator("=")
    print()
    print("  This script demonstrates six classes of attacks against Bitcoin")
    print("  private keys that were generated with insufficient randomness.")
    print("  Each attack corresponds to real historical incidents.")
    print()
    print("  All operations use the real secp256k1 curve (Bitcoin's curve).")
    print("  No actual blockchain interaction occurs -- this is simulation only.")
    print()

    t_total_start = time.time()

    # Run all six parts
    brain_rate = part1_brain_wallet()
    part2_sequential_keys()
    part3_android_securerandom()
    part4_profanity_vanity()
    part5_repeated_r_detection()
    part6_entropy_analysis()

    t_total = time.time() - t_total_start

    # ================================================================
    # Final Summary
    # ================================================================
    print()
    separator("=")
    print("  FINAL SUMMARY")
    separator("=")
    print()
    print("  Six attack classes demonstrated:")
    print()
    print("  1. Brain Wallet:        SHA256(passphrase) -> dictionary attack")
    print("     Effective entropy:    ~20 bits. Recovery: instant.")
    print()
    print("  2. Sequential Keys:     Integers/timestamps as private keys")
    print("     Effective entropy:    ~31 bits. Recovery: seconds (GPU).")
    print()
    print("  3. Android SecureRandom: PRNG with insufficient seeding")
    print("     Effective entropy:    0 bits. Recovery: algebra from 2 sigs.")
    print()
    print("  4. Profanity Generator:  32-bit seed + increment pattern")
    print("     Effective entropy:    32 bits. Recovery: ~4 seconds (GPU).")
    print()
    print("  5. r-Value Reuse:        Same nonce k in multiple signatures")
    print("     Effective entropy:    0 bits. Recovery: one hash map scan.")
    print()
    print("  6. Entropy Comparison:   Full 256-bit vs. all weak methods")
    print("     Proper key requires 3.67 * 10^67 seconds at 10^9/sec.")
    print("     Weak keys require seconds to minutes.")
    print()
    print(f"  Total runtime: {format_time(t_total)}")
    print()

    # Write CSV
    desktop = os.path.expanduser("~/Desktop")
    csv_path = os.path.join(desktop, "weak_rng_brain_wallet.csv")
    if CSV_ROWS:
        fieldnames = [
            "attack_type", "key_source", "entropy_bits",
            "keys_tested", "keys_found", "time_ms", "effective_security",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in CSV_ROWS:
                writer.writerow(row)
        print(f"  CSV report written to: {csv_path}")
        print(f"  Rows: {len(CSV_ROWS)}")
    else:
        print(f"  No CSV data to write.")

    print()
    separator("=")
    print("  Key takeaway: any deviation from 256-bit CSPRNG key generation")
    print("  creates an exploitable weakness. Brain wallets, sequential keys,")
    print("  weak PRNGs, and vanity generators have collectively resulted in")
    print("  hundreds of millions of dollars in cryptocurrency theft.")
    separator("=")
    print()


if __name__ == "__main__":
    main()
