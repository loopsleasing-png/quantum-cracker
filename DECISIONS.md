# Quantum Cracker - Architecture Decision Log

Check here before re-debating a settled choice.

---

## ADR-001: Python as primary language
**Date:** 2026-02-09
**Status:** Accepted
**Context:** Need a language with strong quantum computing and scientific computing libraries.
**Decision:** Python 3.12+ with Qiskit, NumPy, SciPy, Matplotlib.
**Rationale:** De facto standard for quantum computing research. Qiskit provides IBM quantum hardware access. NumPy/SciPy for numerical analysis. Matplotlib for visualization.

## ADR-002: Project structure
**Date:** 2026-02-09
**Status:** Accepted
**Context:** Need organized layout for simulation code, analysis, and experiments.
**Decision:** Single-package `src/quantum_cracker/` layout with `core/`, `analysis/`, `visualization/`, `utils/` subpackages.
**Rationale:** Clean separation of concerns. `src/` layout prevents accidental imports of uninstalled code. Subpackages map to logical domains.

## ADR-003: SciPy sph_harm_y API
**Date:** 2026-02-09
**Status:** Accepted
**Context:** SciPy 1.17 removed `sph_harm` in favor of `sph_harm_y`.
**Decision:** Use `sph_harm_y(l, m, theta, phi)` throughout. Note argument order differs from old API.
**Rationale:** Forward-compatible with modern SciPy. Old API was deprecated.

## ADR-004: PyOpenGL + GLFW for 3D rendering
**Date:** 2026-02-09
**Status:** Accepted
**Context:** Need real-time 3D visualization of voxel grid and threads.
**Decision:** PyOpenGL with GLFW for windowing, OpenGL 3.3 core profile, pyrr for matrix math.
**Rationale:** Stays in Python ecosystem. GLFW works well on macOS with forward-compat flag. Core profile is modern and portable.

## ADR-005: Key-to-grid mapping via spherical harmonics
**Date:** 2026-02-09
**Status:** Accepted
**Context:** Need to map 256 bits into a 78^3 voxel grid in a physically meaningful way.
**Decision:** Map bit index i to SH order (l, m) where bits fill sequentially: l=0 (1 coeff), l=1 (3 coeffs), l=2 (5 coeffs), etc. Bit value (+1/-1) is the coefficient. Sum weighted SH basis on angular grid, modulate by radial Gaussian.
**Rationale:** SH basis is natural for spherical data. 256 bits fill up to l=15 (cumulative 256 = 16^2). The mapping preserves spatial structure -- nearby bits in the key map to similar angular patterns.

## ADR-007: C accelerator via ctypes
**Date:** 2026-02-27
**Status:** Accepted
**Context:** The SQA annealing inner loop is bottlenecked by Python overhead in EC point arithmetic and Ising energy evaluation. Each annealing step makes O(P * N) calls to flip_single/peek_flip with dict iteration.
**Decision:** Implement critical paths in C (`csrc/ec_arith.c`, `csrc/ising_core.c`) compiled to `lib/libqc_accel.so`. Python ctypes wrappers in `src/quantum_cracker/accel/` with transparent fallback to pure Python. Adjacency-list indexing for O(degree) energy deltas instead of O(n_couplings).
**Rationale:** ctypes requires no build step at import time, no Python.h dependency, and graceful degradation. The SQA sweep kernel (`sqa_sweep`) batches all replica updates into one C call, eliminating millions of Python function calls. 128-bit intermediates handle 64-bit primes without overflow.

## ADR-006: Expansion rate as fractional growth
**Date:** 2026-02-09
**Status:** Accepted
**Context:** Using actual Planck length (1.6e-35) as expansion rate caused floating-point precision issues (1 + 1.6e-35 = 1.0 exactly in float64).
**Decision:** Use configurable fractional expansion rate (default 0.01 = 1% per tick). Planck length used only as initial radius.
**Rationale:** Simulation needs observable changes per tick. Physical accuracy maintained by starting at Planck length; rate controls simulation speed.
