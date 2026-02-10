# secp256k1 Security Audit

A comprehensive empirical analysis of all known attack vectors against Bitcoin's secp256k1 elliptic curve. 37 experiments, 63 scripts, 33,000+ lines of Python.

**Author:** KJ M
**License:** AGPL v3 (see LICENSE for dual-licensing details)

## What This Is

The most complete open-source security audit of the secp256k1 elliptic curve discrete logarithm problem (ECDLP). Every known attack class is implemented, demonstrated on small curves (6-16 bit), and analytically projected to 256-bit parameters.

## Results Summary

| Category | Attacks Tested | Result |
|----------|---------------|--------|
| Mathematical (Pohlig-Hellman, Pollard rho, MOV, index calculus, etc.) | 16 | All fail against secp256k1 |
| Implementation (timing, DPA, fault injection, nonce bias, etc.) | 7 | All blocked by libsecp256k1 |
| Quantum (Shor, Grover, hybrid) | 3 | Shor works but needs ~2,330 qubits (est. 2040+) |

**Bottom line:** secp256k1 remains secure against all known classical and near-term quantum attacks when using hardened implementations.

## Project Structure

```
src/quantum_cracker/         # Core library
    core/                    # Key interface, voxel grid, harmonic compiler
    analysis/                # Metrics and statistical validation
    visualization/           # Matplotlib plots + OpenGL 3D renderer
    utils/                   # Constants, math helpers, types

scripts/experiment/          # 63 experiment scripts
    pohlig_hellman_deep_dive.py
    pollard_rho_kangaroo_fixed.py
    mov_pairing_attack.py
    index_calculus_impossibility.py
    semaev_summation_polynomials.py
    weil_descent_ghs.py
    glv_endomorphism_rho.py
    timing_side_channel.py
    differential_power_analysis.py
    ecdsa_fault_injection.py
    nonce_bias_attack.py
    invalid_curve_attack.py
    twist_security_analysis.py
    partial_key_exposure.py
    weak_rng_brain_wallet.py
    shor_algorithm.py
    grover_ecdlp_simulator.py
    grover_hybrid_attack.py
    unified_attack_tree.py
    peer_review_validation.py
    ligo_signal_recovery.py
    harmonic_key_storage.py
    ... and 40+ more

data/                        # Test data (LIGO GW150914, etc.)
PAPER.md                     # Research paper (ready for submission)
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/kjm/quantum-cracker.git
cd quantum-cracker
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
.venv/bin/pytest

# Run a specific experiment
.venv/bin/python scripts/experiment/pohlig_hellman_deep_dive.py
.venv/bin/python scripts/experiment/unified_attack_tree.py

# Run the full simulation
python -m quantum_cracker simulate --random --steps 100 --csv
```

## Requirements

- Python 3.12+
- NumPy, SciPy, Matplotlib
- Qiskit + Qiskit Aer (for quantum experiments)
- scikit-learn (for ML-based experiments)
- PyWavelets (for signal processing benchmarks)

## 24 Attack Vectors Covered

### Mathematical Attacks
1. Pohlig-Hellman decomposition
2. Pollard rho with distinguished points
3. Pollard kangaroo (lambda method)
4. Baby-step giant-step (BSGS)
5. MOV/Frey-Ruck pairing reduction
6. Smart/Satoh-Araki anomalous curve lift
7. Index calculus (impossibility proof)
8. Semaev summation polynomials
9. Weil descent / GHS transfer
10. GLV endomorphism-accelerated rho
11. Multi-target batch DLP
12. Lattice-based HNP reduction

### Implementation Attacks
13. Timing side-channel
14. Differential power analysis (DPA)
15. ECDSA fault injection
16. Nonce bias / lattice recovery
17. Invalid curve point injection
18. Twist security analysis
19. Partial key exposure
20. Weak RNG / brain wallet

### Quantum Attacks
21. Shor's algorithm (ECDLP variant)
22. Grover search acceleration
23. Grover-classical hybrid

### Composition Analysis
24. Unified attack tree (15 attacks as constraints, multi-vector pipelines)

## Citing This Work

If you use this code or data in your research, please cite:

```
KJ M. "A Comprehensive Empirical Analysis of All Known Attack Vectors
Against the secp256k1 Elliptic Curve Discrete Logarithm Problem."
February 2026. https://github.com/kjm/quantum-cracker
```

## Acknowledgments

AI assistance from Claude (Anthropic) was used for code development and experimental design. All scientific content was verified by the human author.

## License

This project is dual-licensed:

- **Open Source:** AGPL v3 -- free for open-source projects
- **Commercial:** Contact KJ M for proprietary use licensing

See [LICENSE](LICENSE) for details.
