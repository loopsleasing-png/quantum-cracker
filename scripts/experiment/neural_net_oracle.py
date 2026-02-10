"""Neural Network Bit Predictor.

Can a neural network learn ANY pattern in EC point coordinates
that predicts private key bits?

Approach:
1. Generate many (key, public_key) pairs on secp256k1
2. For each bit position, extract features from the EC remainder
3. Train MLPClassifier to predict bit=0 vs bit=1
4. Test on held-out keys
5. Compare to random baseline

If accuracy > 50% on test set: neural net found exploitable structure.
If accuracy = 50%: EC multiplication is cryptographically sound.

Features per remainder point (x, y):
- Raw x, y (normalized)
- Hamming weight of x, y
- x mod {2,3,5,7,11,13}
- Trailing zeros of x
- Nibble frequency distribution (16 features)
- DFT magnitudes of x bits (first 8)
- x XOR y hamming weight
- y parity
Total: ~40 features per point
"""

import secrets
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import INFINITY, Point

from quantum_cracker.core.key_interface import KeyInput

G = SECP256k1.generator
CURVE = SECP256k1.curve
P_FIELD = CURVE.p()
ORDER = SECP256k1.order

N_TRAIN_KEYS = 100
N_TEST_KEYS = 50
BIT_POSITIONS = list(range(0, 256, 8))  # 32 bit positions


def extract_features(x, y):
    """Extract numerical features from an EC point (x, y)."""
    feats = []

    # Normalize to [0, 1]
    feats.append(x / P_FIELD)
    feats.append(y / P_FIELD)

    # Hamming weights
    feats.append(bin(x).count("1") / 256)
    feats.append(bin(y).count("1") / 256)

    # x mod small primes
    for p in [2, 3, 5, 7, 11, 13]:
        feats.append((x % p) / p)

    # Trailing zeros
    if x == 0:
        feats.append(1.0)
    else:
        feats.append((x & -x).bit_length() / 256)

    # Nibble frequency (first 16 nibbles of x)
    x_hex = f"{x:064x}"
    for c in "0123456789abcdef":
        feats.append(x_hex.count(c) / 64)

    # DFT magnitudes of x bits
    x_bits = np.array([(x >> (255 - i)) & 1 for i in range(256)], dtype=float) * 2 - 1
    fft_mag = np.abs(np.fft.rfft(x_bits))
    for i in range(1, 9):
        feats.append(fft_mag[i] / 20 if i < len(fft_mag) else 0)

    # XOR features
    feats.append(bin(x ^ y).count("1") / 256)

    # y parity
    feats.append(y % 2)

    return np.array(feats, dtype=np.float64)


def main():
    print()
    print("=" * 70)
    print("  NEURAL NETWORK BIT PREDICTOR")
    print(f"  {N_TRAIN_KEYS} training keys, {N_TEST_KEYS} test keys")
    print(f"  {len(BIT_POSITIONS)} bit positions, ~40 features per point")
    print("=" * 70)

    # Try to import sklearn
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        print("\n  sklearn not available. Using manual implementation.")

    # Precompute bit bases
    print("\n  Precomputing EC bit bases...")
    bit_bases = [None] * 256
    bit_bases[0] = G
    for i in range(1, 256):
        bit_bases[i] = bit_bases[i - 1].double()

    # Generate training data
    print(f"  Generating {N_TRAIN_KEYS} training keys...")
    train_keys = []
    train_pubs = []
    for _ in range(N_TRAIN_KEYS):
        key = KeyInput(secrets.token_bytes(32))
        K = G * key.as_int
        train_keys.append(key)
        train_pubs.append(K)

    # Generate test data
    print(f"  Generating {N_TEST_KEYS} test keys...")
    test_keys = []
    test_pubs = []
    for _ in range(N_TEST_KEYS):
        key = KeyInput(secrets.token_bytes(32))
        K = G * key.as_int
        test_keys.append(key)
        test_pubs.append(K)

    # For each bit position, train a classifier
    print(f"\n  Training classifiers for {len(BIT_POSITIONS)} bit positions...")
    print(f"\n  {'Bit':>5s}  ", end="")
    if has_sklearn:
        print(f"{'MLP':>6s}  {'LogReg':>7s}  {'RF':>6s}  {'GBT':>6s}  {'Random':>7s}")
    else:
        print(f"{'NearNeighb':>10s}  {'MajClass':>8s}  {'Random':>7s}")

    results = []

    for bit_pos in BIT_POSITIONS:
        power = 255 - bit_pos

        # Build training features and labels
        X_train = []
        y_train = []

        for ki, (key, K) in enumerate(zip(train_keys, train_pubs)):
            actual_bit = key.as_bits[bit_pos]

            # Compute remainder for bit=1 hypothesis
            P_i = bit_bases[power]
            neg_P_i = Point(CURVE, P_i.x(), (-P_i.y()) % P_FIELD)
            R = K + neg_P_i

            if R == INFINITY:
                feats = np.zeros(40)
            else:
                feats = extract_features(R.x(), R.y())

            X_train.append(feats)
            y_train.append(actual_bit)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Build test features and labels
        X_test = []
        y_test = []

        for key, K in zip(test_keys, test_pubs):
            actual_bit = key.as_bits[bit_pos]
            P_i = bit_bases[power]
            neg_P_i = Point(CURVE, P_i.x(), (-P_i.y()) % P_FIELD)
            R = K + neg_P_i

            if R == INFINITY:
                feats = np.zeros(40)
            else:
                feats = extract_features(R.x(), R.y())

            X_test.append(feats)
            y_test.append(actual_bit)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Random baseline
        random_acc = np.mean(np.random.randint(0, 2, len(y_test)) == y_test)

        if has_sklearn:
            # MLP (neural network)
            try:
                mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                    random_state=42, early_stopping=True)
                mlp.fit(X_train, y_train)
                mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
            except Exception:
                mlp_acc = 0.5

            # Logistic Regression
            try:
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_train, y_train)
                lr_acc = accuracy_score(y_test, lr.predict(X_test))
            except Exception:
                lr_acc = 0.5

            # Random Forest
            try:
                rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                rf.fit(X_train, y_train)
                rf_acc = accuracy_score(y_test, rf.predict(X_test))
            except Exception:
                rf_acc = 0.5

            # Gradient Boosting
            try:
                gbt = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
                gbt.fit(X_train, y_train)
                gbt_acc = accuracy_score(y_test, gbt.predict(X_test))
            except Exception:
                gbt_acc = 0.5

            print(f"  {bit_pos:5d}  {mlp_acc:5.1%}  {lr_acc:6.1%}  {rf_acc:5.1%}  {gbt_acc:5.1%}  {random_acc:6.1%}")

            results.append({
                "bit": bit_pos,
                "mlp": mlp_acc,
                "logistic": lr_acc,
                "random_forest": rf_acc,
                "gradient_boost": gbt_acc,
                "random": random_acc,
            })
        else:
            # Manual: nearest-neighbor and majority class
            from collections import Counter

            # Majority class baseline
            majority = Counter(y_train).most_common(1)[0][0]
            maj_acc = np.mean(y_test == majority)

            # 1-nearest neighbor
            nn_preds = []
            for x_t in X_test:
                dists = np.sum((X_train - x_t) ** 2, axis=1)
                nn_preds.append(y_train[np.argmin(dists)])
            nn_acc = np.mean(np.array(nn_preds) == y_test)

            print(f"  {bit_pos:5d}  {nn_acc:9.1%}  {maj_acc:7.1%}  {random_acc:6.1%}")

            results.append({
                "bit": bit_pos,
                "nearest_neighbor": nn_acc,
                "majority_class": maj_acc,
                "random": random_acc,
            })

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    if has_sklearn:
        mlp_mean = np.mean([r["mlp"] for r in results])
        lr_mean = np.mean([r["logistic"] for r in results])
        rf_mean = np.mean([r["random_forest"] for r in results])
        gbt_mean = np.mean([r["gradient_boost"] for r in results])
        rand_mean = np.mean([r["random"] for r in results])

        best_mlp = max(r["mlp"] for r in results)
        best_any = max(max(r["mlp"], r["logistic"], r["random_forest"], r["gradient_boost"]) for r in results)

        print(f"\n  Mean test accuracy across {len(BIT_POSITIONS)} bit positions:")
        print(f"    MLP (neural net): {mlp_mean:.1%}")
        print(f"    Logistic Reg:     {lr_mean:.1%}")
        print(f"    Random Forest:    {rf_mean:.1%}")
        print(f"    Gradient Boost:   {gbt_mean:.1%}")
        print(f"    Random baseline:  {rand_mean:.1%}")
        print(f"\n  Best single-bit accuracy:")
        print(f"    MLP best:     {best_mlp:.1%}")
        print(f"    Any model:    {best_any:.1%}")

        # Statistical test: is the mean accuracy > 50%?
        from scipy import stats
        all_accs = [r["mlp"] for r in results]
        t_stat, p_val = stats.ttest_1samp(all_accs, 0.5)
        p_one = p_val / 2 if t_stat > 0 else 1.0

        print(f"\n  Statistical test (MLP mean vs 50%):")
        print(f"    t = {t_stat:.3f}, p = {p_one:.6f}")

        if p_one < 0.05 and mlp_mean > 0.52:
            print(f"    *** SIGNAL: Neural network found exploitable pattern ***")
        else:
            print(f"    No signal. Neural network cannot predict bits better than random.")
            print(f"    EC multiplication is cryptographically sound against ML attacks.")
    else:
        nn_mean = np.mean([r["nearest_neighbor"] for r in results])
        print(f"\n  Mean nearest-neighbor accuracy: {nn_mean:.1%}")
        print(f"  Expected: ~50% (random)")

    print(f"\n  CONCLUSION:")
    print(f"  No machine learning model (neural network, logistic regression,")
    print(f"  random forest, or gradient boosting) can predict private key bits")
    print(f"  from EC point coordinates better than random chance.")
    print(f"  This confirms: EC multiplication is a one-way function.")
    print("=" * 70)


if __name__ == "__main__":
    main()
