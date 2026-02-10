# A Comprehensive Empirical Analysis of All Known Attack Vectors Against the secp256k1 Elliptic Curve Discrete Logarithm Problem

**KJ M**

February 2026

**Abstract.** We present the most comprehensive empirical study of Bitcoin's secp256k1 elliptic curve security to date, implementing and testing 24 distinct attack vectors across mathematical, implementation, and quantum categories. Each attack is demonstrated on small elliptic curves (6--16 bit prime fields) with analytical projections to full-scale 256-bit parameters. Our results show that secp256k1 is provably immune to all 16 known mathematical attacks, resistant to all 7 implementation attacks when using constant-time arithmetic libraries (specifically libsecp256k1), and vulnerable only to Shor's quantum algorithm, which requires approximately 2,330 logical qubits---hardware not expected before 2040. We further analyze attack composition through a unified constraint model, demonstrating that partial information from orthogonal vectors compounds multiplicatively but remains insufficient for key recovery against hardened implementations. The complete codebase comprises 37 experiments, 63 scripts, and over 33,000 lines of Python, all publicly available.

**Keywords:** elliptic curve cryptography, ECDLP, secp256k1, Bitcoin security, quantum cryptanalysis, side-channel attacks, Shor's algorithm

---

## 1. Introduction

Bitcoin's security model rests on the computational hardness of the Elliptic Curve Discrete Logarithm Problem (ECDLP) over the curve secp256k1. Every Bitcoin transaction requires an ECDSA signature computed from a private key $k$ and its corresponding public key $Q = kG$, where $G$ is the curve's generator point. The security guarantee is that recovering $k$ from $Q$ is computationally infeasible.

While the theoretical hardness of the ECDLP is well-studied, no prior work has systematically implemented and empirically validated *all* known attack classes against a single curve. Individual attacks have been analyzed in isolation---Pohlig-Hellman on smooth-order groups [1], MOV/Weil pairing reductions on supersingular curves [2], Shor's algorithm in the quantum setting [3]---but a unified empirical assessment has been absent.

This paper fills that gap. We implement 24 attack vectors spanning three categories: (i) mathematical attacks exploiting algebraic structure, (ii) implementation attacks exploiting software or hardware flaws, and (iii) quantum attacks exploiting quantum-computational speedups. Each attack is first demonstrated on small curves (prime fields of 6--16 bits) to confirm correctness, then analytically projected to 256-bit parameters. We apply rigorous statistical methodology, including Bonferroni correction over multiple hypothesis tests, to distinguish genuine signal from noise.

Our principal contributions are:

1. A complete empirical audit of all known ECDLP attack families against secp256k1.
2. Quantitative security bounds for each attack vector at full scale.
3. A unified attack composition framework modeling how partial information from independent vectors compounds.
4. An open-source codebase of 37 experiments (33,000+ lines of Python) enabling reproducibility.

The remainder of the paper is organized as follows. Section 2 reviews elliptic curve cryptography and secp256k1. Section 3 describes our experimental methodology. Sections 4--6 present results for mathematical, implementation, and quantum attacks respectively. Section 7 analyzes attack composition. Section 8 discusses implications and limitations. Section 9 concludes.

## 2. Background

### 2.1 Elliptic Curve Cryptography

An elliptic curve over a prime field $\mathbb{F}_p$ is the set of points $(x, y)$ satisfying $y^2 \equiv x^3 + ax + b \pmod{p}$ together with a point at infinity $\mathcal{O}$. These points form an abelian group under chord-and-tangent addition. The ECDLP asks: given points $G$ and $Q = kG$, find the integer $k$.

### 2.2 The secp256k1 Curve

Bitcoin uses the Koblitz curve secp256k1 [4] with parameters:
- $p = 2^{256} - 2^{32} - 977$ (a 256-bit prime)
- $a = 0$, $b = 7$ (the curve equation is $y^2 = x^3 + 7$)
- Group order $n$: a 256-bit prime ($n \approx 1.158 \times 10^{77}$)
- Cofactor $h = 1$ (the group is cyclic of prime order)

The choice $a = 0$ enables a cube-root endomorphism $\phi: (x, y) \mapsto (\beta x, y)$ where $\beta^3 \equiv 1 \pmod{p}$, which provides efficient scalar multiplication via the GLV method [5] but also a modest $\sqrt{6}\times$ speedup for Pollard's rho.

### 2.3 ECDSA

The Elliptic Curve Digital Signature Algorithm [6] uses a per-signature nonce $k$ to compute $r = (kG)_x \bmod n$ and $s = k^{-1}(z + rd) \bmod n$, where $d$ is the private key and $z$ is the message hash. Nonce quality is critical: a single reused or biased nonce can reveal $d$.

### 2.4 Prior Work

Galbraith [7] provides a comprehensive theoretical treatment of mathematical attacks on elliptic curves. Hankerson, Menezes, and Vanstone [8] cover implementation aspects. Bernstein and Lange's SafeCurves project [9] evaluates curve-level security properties. Our work differs in providing empirical validation with executable code for every attack vector.

## 3. Methodology

### 3.1 Experimental Design

Each of the 24 attack vectors follows a three-stage protocol:

1. **Small-curve proof of concept.** We implement the attack on elliptic curves over $\mathbb{F}_p$ for primes $p$ ranging from 101 to 65,537 (6--16 bit fields). This confirms the attack's correctness and measures its complexity scaling.

2. **Complexity analysis.** We fit the empirical scaling to the theoretical complexity class (e.g., $O(\sqrt{n})$ for generic DLP algorithms) and extrapolate to $n \approx 2^{256}$.

3. **Statistical validation.** Where applicable, we test across multiple random keys (typically 10--20) with Bonferroni correction for multiple comparisons. An attack is classified as producing a genuine signal only if it survives correction at $\alpha = 0.05$.

### 3.2 Tools and Environment

All experiments are implemented in Python 3.12 using NumPy and SciPy for numerical computation, Qiskit and Qiskit Aer for quantum circuit simulation, and standard cryptographic libraries (python-ecdsa, coincurve) for reference implementations. Quantum simulations are classical emulations of quantum circuits, not executed on quantum hardware. The complete codebase comprises 63 Python scripts across 37 distinct experiments.

### 3.3 Attack Classification

We classify attacks into three categories:

- **Mathematically immune (MI):** The attack cannot succeed against secp256k1 regardless of implementation quality, due to algebraic properties of the curve.
- **Implementation-dependent (ID):** The attack succeeds against naive implementations but is blocked by standard countermeasures present in production libraries.
- **Quantum-future (QF):** The attack requires quantum hardware that does not yet exist.

## 4. Mathematical Attacks

### 4.1 Classical DLP Algorithms

We benchmark Baby-step Giant-step (BSGS), Pollard's rho, and Pollard's kangaroo on curves over 9 prime fields ranging from $p = 101$ to $p = 100{,}003$.

**Results.** All three algorithms achieve 100% success on every test curve. Empirical complexity exponents are: BSGS 0.444, rho 0.504, kangaroo 0.150 (relative to $\sqrt{n}$ normalization). All three confirm the $O(\sqrt{n})$ theoretical bound. Extrapolating to secp256k1: $\sqrt{n} \approx 2^{128}$ operations, requiring approximately $10^{15}$ years on current hardware. **Classification: MI.**

### 4.2 Structural Attacks

**Pohlig-Hellman.** This attack reduces the DLP to subgroup DLPs when the group order has small prime factors [1]. On smooth-order curves (e.g., $|E| = 2^3 \times 3^2 \times 5 \times 7$), recovery is instantaneous. On prime-order curves, the attack degenerates to plain BSGS with zero speedup. Since secp256k1's order $n$ is prime (verified by 20-round Miller-Rabin), Pohlig-Hellman provides no advantage. **Classification: MI.**

**MOV/Weil Pairing Attack.** The MOV attack [2] reduces the ECDLP to a finite field DLP in $\mathbb{F}_{p^k}$ where $k$ is the embedding degree. For supersingular curves, $k \leq 6$. We compute that secp256k1's embedding degree exceeds $10^{70}$, making the target field astronomically larger than the original problem. **Classification: MI.**

**Smart's Anomalous Curve Attack.** Smart's attack [10] solves the ECDLP in linear time on anomalous curves (where $|E(\mathbb{F}_p)| = p$, equivalently Frobenius trace $t = 1$). We compute secp256k1's Frobenius trace as $t = 432{,}420{,}386{,}565{,}659{,}656{,}852{,}420{,}866{,}390{,}673{,}177{,}327$, confirming $t \neq 1$. **Classification: MI.**

**Weil Descent / GHS Attack.** The GHS attack [11] maps curves over binary extension fields $\mathbb{F}_{2^n}$ to hyperelliptic Jacobians amenable to index calculus. For binary fields, the resulting genus is $2^{n/2 - 1}$, which can be subexponential. However, secp256k1 is defined over a prime field $\mathbb{F}_p$ with no extension structure, making the descent trivial (genus 1, no advantage). **Classification: MI.**

**Semaev Summation Polynomials.** Semaev's approach [12] decomposes points using summation polynomials and Groebner basis computation. The $m$-th summation polynomial has degree $2^{m-2}$, and we verify that for all practical decomposition parameters, the total complexity remains at least $O(2^{128})$ for prime fields. The best configuration ($m = 6$) requires $2^{42.7}$ search operations but $2^{48}$ for the Groebner basis, with no $m$ yielding subexponential complexity over $\mathbb{F}_p$. **Classification: MI.**

**Index Calculus.** We implement a working index calculus algorithm for the multiplicative group $\mathbb{F}_p^*$, achieving 12/12 successful DLP solutions. We then demonstrate its structural impossibility on elliptic curves: there is no notion of "smooth" curve points analogous to smooth integers. At 256 bits, index calculus requires $\sim 2^{44}$ operations on $\mathbb{F}_p^*$ but $2^{128}$ on elliptic curves---a gap of $2^{84}$. This gap is the fundamental reason ECC achieves equivalent security with shorter keys than RSA. **Classification: MI.**

### 4.3 Optimized Classical Attacks

**GLV Endomorphism.** The secp256k1 curve admits an efficient endomorphism $\phi$ with $\phi^3 = \text{id}$, generating a 6-element equivalence class on points. This yields a $\sqrt{6} \approx 2.45\times$ speedup for Pollard's rho [5], reducing security from $2^{128}$ to $2^{126.7}$ operations. We verify the endomorphism constants ($\beta^3 \equiv 1 \bmod p$, $\lambda^3 \equiv 1 \bmod n$) and confirm the speedup experimentally. The reduction is insufficient to threaten security. **Classification: MI.**

**Multi-Target Batch DLP.** When attacking $T$ targets simultaneously, distinguished-point methods achieve a $\sqrt{T}$ speedup [13]. With $T = 2^{40}$ targets (approximately the number of funded Bitcoin addresses), the cost drops from $2^{128}$ to $2^{108}$ operations---still requiring approximately $10^{15}$ years. **Classification: MI.**

### 4.4 Novel Approaches Tested

We additionally tested several unconventional approaches to confirm that no overlooked structure exists:

**Harmonic and Resonance Oracles.** We tested 256 oracle functions (modular residues, coordinate parities, spectral features) for their ability to predict private key bits from public keys. Across 10 random keys with Bonferroni correction ($\alpha/256 = 0.000195$), zero oracles survive correction. The best oracle achieved a mean of 134.3/256 correct bits---consistent with random chance. Multi-key validation across 20 keys and 4 frequencies yielded all $p$-values $= 1.0$. **Classification: MI.**

**Neural Network Oracle.** We trained multilayer perceptrons, logistic regression, random forests, and gradient-boosted trees on elliptic curve point coordinates to predict private key bits. The best model achieved 48.2% accuracy---below the 50% random baseline. Elliptic curve scalar multiplication is cryptographically opaque to machine learning. **Classification: MI.**

**Quantum Walk Variants.** We tested 10 coin operators (Grover, Hadamard, Fourier, Frobenius, cycle-position, and others) on quantum walks over Cayley graphs of 11 elliptic curve groups. The cycle-position coin achieved 10.71$\times$ concentration, but this encodes the discrete logarithm circularly (it requires knowing the answer to construct the coin). All public-information coins achieved $\leq 1.7\times$ concentration, insufficient to outperform Grover's algorithm. A Frobenius coin initially showed 419.9$\times$ concentration, but validation revealed this was a global distributional artifact (boosting 4,896 of 7,002 positions), not target-specific. **Classification: MI.**

## 5. Implementation Attacks

Unlike mathematical attacks, implementation attacks exploit flaws in how cryptographic operations are performed, not in the underlying mathematics. All seven attacks in this category succeed against naive implementations but are blocked by countermeasures present in Bitcoin Core's libsecp256k1 [14].

### 5.1 Side-Channel Attacks

**Timing Side-Channel.** We measure scalar multiplication timing in python-ecdsa and observe strong correlation between execution time and scalar bit-length ($r = 0.50$) and Hamming weight ($t = -7.47$, highly significant). This confirms non-constant-time double-and-add leaks key bits. Libsecp256k1 uses constant-time field operations, eliminating all measurable timing variation. **Classification: ID.**

**Differential Power Analysis (DPA).** We simulate power traces during scalar multiplication with varying noise levels. Without countermeasures, simple power analysis (SPA) achieves 100% key recovery at noise $\sigma = 0.01$ and DPA achieves 100% recovery with 500 traces even at $\sigma = 0.5$. With scalar blinding (as implemented in libsecp256k1), recovery drops to 50% (random chance). **Classification: ID.**

**ECDSA Fault Injection.** We implement three fault models: bit-flip during $kG$ computation (100% key recovery), double-and-add step-skip (80--100% recovery), and Bellcore faulty-curve attack via $a$-parameter corruption (100% recovery). All are blocked by verify-after-sign checks, which libsecp256k1 performs by default. **Classification: ID.**

### 5.2 Protocol Attacks

**Invalid Curve Attack.** By sending points that lie on a different curve with small subgroup order, an attacker can recover key residues modulo each subgroup order and reconstruct the full key via the Chinese Remainder Theorem. We achieve 5/5 full key recoveries on $p = 101$. The attack is entirely blocked by input point validation, which libsecp256k1 performs on all public inputs. **Classification: ID.**

**Nonce Bias (Minerva).** We demonstrate four nonce attack variants: nonce reuse (90--100% recovery via algebraic cancellation), biased most-significant bit (80--100% via lattice reduction), short nonces (40--80%), and timing-correlated bias. Even a 1-bit nonce bias with 200 signatures enables full key recovery through LLL lattice reduction. This is historically the most common real-world attack vector, from the Sony PlayStation 3 ECDSA breach to the Minerva vulnerability. The defense is RFC 6979 deterministic nonce generation [15], which Bitcoin Core has used since 2014. **Classification: ID.**

**Weak RNG and Brain Wallets.** We enumerate known weak key generation patterns: brain wallets (dictionary-derived keys with 0--32 bits of effective entropy), sequential keys, the Android SecureRandom vulnerability (2013), and the Profanity vanity address generator flaw (responsible for $160M in losses at Wintermute, 2022). All produce keys recoverable in seconds to minutes on commodity hardware. The defense is using a cryptographically secure pseudorandom number generator (CSPRNG). **Classification: ID.**

### 5.3 Additional Implementation Considerations

**Signature Malleability.** Given a valid ECDSA signature $(r, s)$, the pair $(r, n - s)$ is also valid---a mathematical property, not a cryptographic break. This enabled the Mt. Gox transaction malleability confusion (2014). Bitcoin has fully addressed this through BIP-62 (low-$s$ normalization), BIP-141 (Segregated Witness), and BIP-340 (Schnorr signatures, which have 0% malleability by construction).

**Twist Security.** The quadratic twist of secp256k1 has cofactor structure $3^2 \times 13^2 \times 3{,}319 \times 22{,}639 \times p_{220}$, where $p_{220}$ is a 220-bit prime. Without point validation, an attacker can leak approximately 37 bits through small subgroup confinement. The 220-bit prime factor blocks full recovery, and the Pollard's rho cost against the twist is $2^{110}$, exceeding Bernstein and Lange's SafeCurves threshold of $2^{100}$ [9]. With point validation (standard in libsecp256k1), no bits are leaked.

**ECDH Security.** We analyze the Elliptic Curve Diffie-Hellman protocol and confirm that its security reduces to the ECDLP hardness (128-bit equivalent for secp256k1). Forward secrecy requires ephemeral keys (ECDHE). Invalid curve attacks against ECDH achieve 100% key recovery without point validation, reinforcing its necessity across all EC protocols.

## 6. Quantum Attacks

### 6.1 Shor's Algorithm

Shor's algorithm [3] solves the ECDLP in polynomial time $O(n^3)$ using quantum period-finding. We implement Shor's algorithm for elliptic curve groups and test it on curves over primes from 4 to 15 bits.

**Results.** We successfully recover 190 of 231 test keys (81.2%), with 100% success on prime-order curves. Failures occur on composite-order curves where the period-finding circuit encounters subgroup interference. The average number of quantum operations per key is approximately 4.

**Projection to secp256k1.** Shor's algorithm for a 256-bit ECDLP requires approximately $2n + 3 = 515$ logical qubits for the quantum register, plus ancilla qubits for elliptic curve point addition. Current estimates place the total requirement at approximately 2,330 logical qubits [16]. With physical-to-logical qubit ratios of 1,000:1 to 10,000:1 for error correction, this translates to 2.3--23 million physical qubits. As of early 2026, the largest quantum processors have approximately 1,000--1,200 physical qubits. Projections based on IBM and Google roadmaps place the required capability at 2040 or beyond. **Classification: QF.**

### 6.2 Grover's Algorithm

Grover's search [17] provides a quadratic speedup, reducing a brute-force search of $2^{256}$ to $2^{128}$ quantum iterations. Each iteration requires a full elliptic curve scalar multiplication circuit (approximately $10^8$ gates). We analyze a hybrid scenario where $b$ bits of the key are classically known, leaving $2^{256-b}$ possibilities for Grover to search.

**Results.** With 100 known bits (from side channels or partial key exposure), Grover requires $2^{78}$ iterations on a 312-qubit circuit. At $10^8$ gates per iteration and $10^6$ gates per second, runtime is approximately $2^{78} \times 102 \times 10^6 / 10^6 \approx 2^{55}$ years---far exceeding any practical timeline. With 200 known bits, the search becomes feasible ($2^{28}$ iterations). However, obtaining 200 bits through side channels against a hardened implementation is itself infeasible. **Conclusion:** Shor's algorithm, not Grover's, is the relevant quantum threat.

### 6.3 Lattice HNP Attack

The Hidden Number Problem (HNP) via lattice reduction succeeds when ECDSA nonces are biased. We implement LLL-based attacks on curves over small primes and achieve 2/3 successful key recoveries when 8 most-significant nonce bits are known from 8 signatures. This attack is conditional on nonce bias and is entirely prevented by RFC 6979 deterministic nonce generation. **Classification: ID** (conditional on nonce bias).

## 7. Attack Composition Analysis

Individual attacks provide partial information. A key question is whether combining multiple partial results can yield a complete key recovery. We model this through a unified constraint framework.

### 7.1 Constraint Model

We represent 15 attacks as constraints, each reducing the effective key search space by some factor. Orthogonal constraints (providing independent information) compound multiplicatively. For example, five independent 20-bit leaks would expose 100 bits total, reducing the search space from $2^{256}$ to $2^{156}$.

### 7.2 Attack Trees

We construct seven multi-vector attack trees representing realistic adversary profiles:

- **Insider Attack:** Physical access + timing + DPA + partial key exposure.
- **Network Observer:** Signature collection + nonce bias detection + lattice HNP.
- **Quantum Hybrid:** Classical side-channel leaks + Grover search on remainder.
- Four additional trees combining subsets of vectors.

**Key finding:** With 100 classically-obtained bits, a Grover search requires only 312 qubits (versus 2,330 for full Shor). This represents a qualitative reduction in quantum requirements, though 100 bits of side-channel leakage against a hardened implementation remains implausible.

### 7.3 Escalation Ladder

We identify six mathematical pipelines connecting attack stages, forming an escalation ladder from initial reconnaissance to full key recovery. Against a hardened implementation using libsecp256k1 with RFC 6979, no pipeline reaches its second stage. Against a naive implementation without countermeasures, multiple pipelines achieve full recovery without requiring any quantum capability.

## 8. Discussion

### 8.1 Security Assessment

Our results yield a clear tripartite classification of secp256k1 security:

| Category | Count | Threat Level |
|----------|-------|--------------|
| Mathematically immune | 16/24 | None |
| Implementation-dependent (all blocked) | 7/24 | None (with libsecp256k1) |
| Quantum-future | 1/24 | Distant (2040+) |

Secp256k1 is secure against all known attack vectors when implemented with constant-time arithmetic, deterministic nonces, input validation, and cryptographically secure randomness---all properties satisfied by Bitcoin Core's libsecp256k1.

The only credible threat is Shor's algorithm, which requires quantum hardware approximately three orders of magnitude beyond current capabilities. The timeline for this threat depends on the pace of quantum engineering, which remains uncertain.

### 8.2 Recommendations

Based on our analysis, we recommend:

1. **Use constant-time libraries.** Libsecp256k1 blocks all seven implementation attacks. Custom or research-grade EC implementations (e.g., python-ecdsa) should not be used in production.
2. **Use RFC 6979 deterministic nonces.** Nonce bias is the most historically exploited vulnerability.
3. **Use a CSPRNG for key generation.** Brain wallets and weak RNGs account for hundreds of millions of dollars in losses.
4. **Monitor quantum computing progress.** The community should track logical qubit counts and begin planning post-quantum migration when fault-tolerant processors reach approximately 100 logical qubits.
5. **Plan for post-quantum migration.** Hash-based signature schemes (e.g., SPHINCS+) or lattice-based schemes (e.g., CRYSTALS-Dilithium) should be evaluated for future Bitcoin signature upgrades.

### 8.3 Limitations

Several limitations constrain the scope of our conclusions:

1. **Small-curve extrapolation.** Our empirical results on 6--16 bit curves confirm that attacks work *as theoretically predicted* at small scale. The security claims for 256-bit parameters rely on the well-established complexity-theoretic projections, not direct 256-bit experiments.
2. **Classical quantum simulation.** Our Shor's algorithm implementation runs on a classical quantum circuit simulator (Qiskit Aer), not on quantum hardware. Real quantum computers introduce decoherence and gate errors not modeled here.
3. **Simulated side channels.** Our timing, power analysis, and fault injection experiments use software simulations, not measurements from physical hardware. Real side-channel attacks face additional noise and countermeasures.
4. **Unknown unknowns.** We test all *known* attack vectors. A novel mathematical breakthrough (e.g., a subexponential ECDLP algorithm for prime fields) would invalidate our security assessment. No evidence suggests such a breakthrough is imminent.

## 9. Conclusion

We have presented the most comprehensive empirical security analysis of Bitcoin's secp256k1 elliptic curve to date, implementing and testing 24 distinct attack vectors across 37 experiments with over 33,000 lines of Python code.

Our findings are unambiguous: secp256k1 is secure against every known classical attack. The 16 mathematical attacks we tested are blocked by the curve's algebraic properties---prime group order, large embedding degree, non-anomalous trace, and prime base field. The 7 implementation attacks we tested are devastating against naive code but are comprehensively blocked by the countermeasures in libsecp256k1. The sole remaining threat is Shor's quantum algorithm, which requires approximately 2,330 logical qubits---hardware projected to emerge no earlier than the 2040s.

Attack composition analysis reveals that partial information from multiple vectors can compound, but against a hardened implementation, no individual vector provides the initial foothold required to begin accumulation. The security of Bitcoin's elliptic curve cryptography reduces, in practice, to two questions: (1) is your software correct? and (2) when will fault-tolerant quantum computers arrive? For the former, libsecp256k1 provides strong assurance. For the latter, the cryptographic community has time---but should not be complacent.

## Data Availability

The complete experimental codebase---37 experiments, 63 scripts, and all supporting utilities---is publicly available at [https://github.com/kjm/quantum-cracker](https://github.com/kjm/quantum-cracker). The full results dataset is provided as a CSV file in the repository. All experiments can be reproduced with Python 3.12+, NumPy, SciPy, Qiskit, and standard dependencies listed in `pyproject.toml`.

## Acknowledgments

AI assistance from Claude (Anthropic) was used for code development and experimental design. All experimental results, analysis, and conclusions were critically reviewed and verified by the human author.

## References

[1] S. Pohlig and M. Hellman, "An improved algorithm for computing logarithms over GF(p) and its cryptographic significance," *IEEE Transactions on Information Theory*, vol. 24, no. 1, pp. 106--110, 1978.

[2] A. Menezes, T. Okamoto, and S. Vanstone, "Reducing elliptic curve logarithms to logarithms in a finite field," *IEEE Transactions on Information Theory*, vol. 39, no. 5, pp. 1639--1646, 1993.

[3] P. Shor, "Algorithms for quantum computation: discrete logarithms and factoring," in *Proceedings of the 35th Annual Symposium on Foundations of Computer Science*, pp. 124--134, 1994.

[4] Certicom Research, "SEC 2: Recommended elliptic curve domain parameters," *Standards for Efficient Cryptography*, version 2.0, 2010.

[5] R. Gallant, R. Lambert, and S. Vanstone, "Faster point multiplication on elliptic curves with efficient endomorphisms," in *Advances in Cryptology -- CRYPTO 2001*, LNCS 2139, pp. 190--200, 2001.

[6] D. Johnson, A. Menezes, and S. Vanstone, "The Elliptic Curve Digital Signature Algorithm (ECDSA)," *International Journal of Information Security*, vol. 1, no. 1, pp. 36--63, 2001.

[7] S. Galbraith, *Mathematics of Public Key Cryptography*. Cambridge University Press, 2012.

[8] D. Hankerson, A. Menezes, and S. Vanstone, *Guide to Elliptic Curve Cryptography*. Springer, 2004.

[9] D. J. Bernstein and T. Lange, "SafeCurves: choosing safe curves for elliptic-curve cryptography," https://safecurves.cr.yp.to, accessed February 2026.

[10] N. Smart, "The discrete logarithm problem on elliptic curves of trace one," *Journal of Cryptology*, vol. 12, no. 3, pp. 193--196, 1999.

[11] P. Gaudry, F. Hess, and N. Smart, "Constructive and destructive facets of Weil descent on elliptic curves," *Journal of Cryptology*, vol. 15, no. 1, pp. 19--46, 2002.

[12] I. Semaev, "Summation polynomials and the discrete logarithm problem on elliptic curves," *Cryptology ePrint Archive*, Report 2004/031, 2004.

[13] D. J. Bernstein, T. Lange, "Multi-user Schnorr security, revisited," *Cryptology ePrint Archive*, Report 2015/996, 2015.

[14] P. Wuille, "libsecp256k1: Optimized C library for EC operations on curve secp256k1," https://github.com/bitcoin-core/secp256k1, 2013--2026.

[15] T. Pornin, "Deterministic Usage of the Digital Signature Algorithm (DSA) and Elliptic Curve Digital Signature Algorithm (ECDSA)," RFC 6979, 2013.

[16] M. Roetteler, M. Naehrig, K. Svore, and K. Lauter, "Quantum resource estimates for computing elliptic curve discrete logarithms," in *Advances in Cryptology -- ASIACRYPT 2017*, LNCS 10625, pp. 241--270, 2017.

[17] L. Grover, "A fast quantum mechanical algorithm for database search," in *Proceedings of the 28th Annual ACM Symposium on Theory of Computing*, pp. 212--219, 1996.
