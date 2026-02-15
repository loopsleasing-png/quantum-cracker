"""Parity Hamiltonian for the EC discrete logarithm problem.

Constructs an Ising Hamiltonian H = H_constraint + H_parity where:
- H_constraint penalizes spin configurations that don't satisfy k*G = P
- H_parity implements the PDQM parity energy gap and nearest-neighbor coupling

The ground state of H encodes the private key k.

For small curves (N <= 20): uses exact 2^N diagonal penalty.
For large curves (N > 20): uses ECEnergyEvaluator for O(1) point-
addition constraint evaluation per spin flip.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from quantum_cracker.parity.ec_constraints import (
    ECConstraintEncoder,
    ECEnergyEvaluator,
    SmallEC,
)
from quantum_cracker.parity.types import ParityConfig


class ParityHamiltonian:
    """Ising Hamiltonian whose ground state is the EC private key.

    Two modes of operation:
    - N <= 20: Full 2^N dense matrix with exact diagonalization.
    - N > 20: MCMC-only mode using ECEnergyEvaluator for constraint
      evaluation via incremental point addition.
    """

    def __init__(self, config: ParityConfig) -> None:
        self.config = config
        self.n_spins = config.n_spins
        self._matrix: NDArray[np.float64] | None = None
        self._eigenvalues: NDArray[np.float64] | None = None
        self._eigenvectors: NDArray[np.float64] | None = None
        self._constraint_diagonal: NDArray[np.float64] | None = None
        self._h_fields: NDArray[np.float64] | None = None
        self._j_couplings: dict[tuple[int, int], float] = {}
        self._target_spins: NDArray[np.int8] | None = None
        self._ec_evaluator: ECEnergyEvaluator | None = None

    @property
    def uses_ec_evaluator(self) -> bool:
        """True if using direct EC evaluation instead of diagonal."""
        return self._ec_evaluator is not None

    @classmethod
    def from_ec_dlp(
        cls,
        curve: SmallEC,
        generator: tuple[int, int],
        public_key: tuple[int, int],
        config: ParityConfig | None = None,
    ) -> ParityHamiltonian:
        """Build Hamiltonian where the ground state is the private key.

        For N <= 20: exact penalty diagonal (2^N entries).
        For N > 20: ECEnergyEvaluator with precomputed power points.

        The constraint weight is automatically scaled to dominate
        the parity terms: W = max(config.constraint_weight, 2 * N * delta).
        """
        n_bits = curve.key_bit_length()
        if config is None:
            config = ParityConfig(n_spins=n_bits)
        else:
            config.n_spins = n_bits

        # Ensure constraint weight dominates parity terms
        parity_energy_scale = n_bits * (config.delta_e + config.j_coupling)
        config.constraint_weight = max(
            config.constraint_weight, 2.0 * parity_energy_scale
        )

        h = cls(config)
        encoder = ECConstraintEncoder(curve, generator, public_key)

        if n_bits <= 20:
            h._constraint_diagonal = encoder.spin_penalty_diagonal()
        else:
            h._ec_evaluator = encoder.make_evaluator()

        h._build_parity_terms()
        return h

    @classmethod
    def from_known_key(
        cls,
        key_bits: list[int],
        config: ParityConfig | None = None,
    ) -> ParityHamiltonian:
        """Build Hamiltonian from a known key (for validation)."""
        n = len(key_bits)
        if config is None:
            config = ParityConfig(n_spins=n)
        else:
            config.n_spins = n

        h = cls(config)
        h._target_spins = np.array([1 - 2 * b for b in key_bits], dtype=np.int8)

        if n <= 20:
            size = 1 << n
            h._constraint_diagonal = np.ones(size, dtype=np.float64)
            target_idx = 0
            for j in range(n):
                if key_bits[j] == 1:
                    target_idx |= 1 << j
            h._constraint_diagonal[target_idx] = 0.0

        h._h_fields = np.array(
            [0.1 * (1 - 2 * b) for b in key_bits], dtype=np.float64
        )

        h._build_parity_terms()
        return h

    def _build_parity_terms(self) -> None:
        """Add PDQM parity Ising terms.

        H_parity = -Delta/2 * sum_i(sigma_i) - J * sum_{<i,j>}(sigma_i * sigma_j)
        """
        n = self.n_spins
        delta = self.config.delta_e
        j_coupling = self.config.j_coupling

        if self._h_fields is None:
            self._h_fields = np.zeros(n, dtype=np.float64)
        self._h_fields += -delta / 2.0

        if self.config.coupling_topology == "chain":
            for i in range(n - 1):
                key = (i, i + 1)
                self._j_couplings[key] = (
                    self._j_couplings.get(key, 0.0) - j_coupling
                )
        elif self.config.coupling_topology == "all_to_all":
            scaled_j = j_coupling / n
            for i in range(n):
                for j in range(i + 1, n):
                    key = (i, j)
                    self._j_couplings[key] = (
                        self._j_couplings.get(key, 0.0) - scaled_j
                    )

    # -- Dense matrix methods (N <= 20) --

    def to_matrix(self) -> NDArray[np.float64]:
        """Full dense Hamiltonian. Only feasible for N <= 20."""
        n = self.n_spins
        if n > 20:
            raise ValueError(f"N={n} too large for dense matrix (max 20)")
        if self._matrix is not None:
            return self._matrix

        size = 1 << n
        H = np.zeros((size, size), dtype=np.float64)

        if self._constraint_diagonal is not None:
            np.fill_diagonal(
                H, self.config.constraint_weight * self._constraint_diagonal
            )

        for idx in range(size):
            spins = self._index_to_spins(idx, n)
            if self._h_fields is not None:
                H[idx, idx] += np.dot(self._h_fields, spins)
            for (i, j), j_val in self._j_couplings.items():
                H[idx, idx] += j_val * spins[i] * spins[j]

        self._matrix = H
        return H

    def ground_state(self) -> tuple[NDArray[np.float64], float]:
        """Exact ground state via diagonalization (N <= 20)."""
        H = self.to_matrix()
        if self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors = scipy.linalg.eigh(H)
        return self._eigenvectors[:, 0], float(self._eigenvalues[0])

    def ground_state_spins(self) -> NDArray[np.int8]:
        """Spin configuration of the ground state (N <= 20)."""
        H = self.to_matrix()
        diag = np.diag(H)
        gs_idx = int(np.argmin(diag))
        return self._index_to_spins(gs_idx, self.n_spins)

    def ground_state_key(self) -> int:
        """Private key from the ground state (N <= 20)."""
        return self.spins_to_key(self.ground_state_spins())

    # -- Energy evaluation (any N) --

    def energy(self, spins: NDArray[np.int8]) -> float:
        """Evaluate H(sigma) for a given spin configuration."""
        e = 0.0

        # Constraint term
        if self._constraint_diagonal is not None:
            idx = self._spins_to_index(spins, self.n_spins)
            e += self.config.constraint_weight * self._constraint_diagonal[idx]
        elif self._ec_evaluator is not None:
            self._ec_evaluator.set_state_from_spins(spins)
            e += self.config.constraint_weight * self._ec_evaluator.constraint_penalty()

        # Local fields
        if self._h_fields is not None:
            e += float(np.dot(self._h_fields, spins))

        # Couplings
        for (i, j), j_val in self._j_couplings.items():
            e += j_val * spins[i] * spins[j]

        return e

    def energy_change_single_flip(
        self, spins: NDArray[np.int8], flip_idx: int
    ) -> float:
        """Energy change from flipping one spin."""
        s_i = spins[flip_idx]
        dE = 0.0

        if self._h_fields is not None:
            dE += -2.0 * self._h_fields[flip_idx] * s_i

        for (i, j), j_val in self._j_couplings.items():
            if i == flip_idx:
                dE += -2.0 * j_val * s_i * spins[j]
            elif j == flip_idx:
                dE += -2.0 * j_val * spins[i] * s_i

        # Constraint term
        if self._constraint_diagonal is not None:
            n = self.n_spins
            idx_before = self._spins_to_index(spins, n)
            flipped = spins.copy()
            flipped[flip_idx] *= -1
            idx_after = self._spins_to_index(flipped, n)
            dE += self.config.constraint_weight * (
                self._constraint_diagonal[idx_after]
                - self._constraint_diagonal[idx_before]
            )
        elif self._ec_evaluator is not None:
            old_pen = self._ec_evaluator.constraint_penalty()
            new_pen = self._ec_evaluator.peek_flip_single(flip_idx)
            dE += self.config.constraint_weight * (new_pen - old_pen)

        return dE

    def energy_change_pair_flip(
        self, spins: NDArray[np.int8], idx_a: int, idx_b: int
    ) -> float:
        """Energy change from flipping two spins (parity-preserving)."""
        s_a = spins[idx_a]
        s_b = spins[idx_b]
        dE = 0.0

        if self._h_fields is not None:
            dE += -2.0 * self._h_fields[idx_a] * s_a
            dE += -2.0 * self._h_fields[idx_b] * s_b

        for (i, j), j_val in self._j_couplings.items():
            if i == idx_a and j == idx_b:
                pass  # both flip: sigma_a*sigma_b unchanged
            elif i == idx_a:
                dE += -2.0 * j_val * s_a * spins[j]
            elif j == idx_a:
                dE += -2.0 * j_val * spins[i] * s_a
            elif i == idx_b:
                dE += -2.0 * j_val * s_b * spins[j]
            elif j == idx_b:
                dE += -2.0 * j_val * spins[i] * s_b

        # Constraint term
        if self._constraint_diagonal is not None:
            n = self.n_spins
            idx_before = self._spins_to_index(spins, n)
            flipped = spins.copy()
            flipped[idx_a] *= -1
            flipped[idx_b] *= -1
            idx_after = self._spins_to_index(flipped, n)
            dE += self.config.constraint_weight * (
                self._constraint_diagonal[idx_after]
                - self._constraint_diagonal[idx_before]
            )
        elif self._ec_evaluator is not None:
            old_pen = self._ec_evaluator.constraint_penalty()
            new_pen = self._ec_evaluator.peek_flip_pair(idx_a, idx_b)
            dE += self.config.constraint_weight * (new_pen - old_pen)

        return dE

    def apply_single_flip(self, spins: NDArray[np.int8], flip_idx: int) -> None:
        """Apply a single spin flip and update the EC evaluator state."""
        spins[flip_idx] *= -1
        if self._ec_evaluator is not None:
            self._ec_evaluator.flip_single(flip_idx)

    def apply_pair_flip(
        self, spins: NDArray[np.int8], idx_a: int, idx_b: int
    ) -> None:
        """Apply a pair spin flip and update the EC evaluator state."""
        spins[idx_a] *= -1
        spins[idx_b] *= -1
        if self._ec_evaluator is not None:
            self._ec_evaluator.flip_single(idx_a)
            self._ec_evaluator.flip_single(idx_b)

    def sync_evaluator(self, spins: NDArray[np.int8]) -> None:
        """Sync the EC evaluator to match the current spin state."""
        if self._ec_evaluator is not None:
            self._ec_evaluator.set_state_from_spins(spins)

    # -- Conversion utilities --

    @staticmethod
    def spins_to_key(spins: NDArray[np.int8]) -> int:
        key = 0
        for j, s in enumerate(spins):
            if s == -1:
                key |= 1 << j
        return key

    @staticmethod
    def key_to_spins(key: int, n_bits: int) -> NDArray[np.int8]:
        spins = np.ones(n_bits, dtype=np.int8)
        for j in range(n_bits):
            if (key >> j) & 1:
                spins[j] = -1
        return spins

    @staticmethod
    def _index_to_spins(idx: int, n: int) -> NDArray[np.int8]:
        spins = np.ones(n, dtype=np.int8)
        for j in range(n):
            if (idx >> j) & 1:
                spins[j] = -1
        return spins

    @staticmethod
    def _spins_to_index(spins: NDArray[np.int8], n: int) -> int:
        idx = 0
        for j in range(n):
            if spins[j] == -1:
                idx |= 1 << j
        return idx

    @property
    def ising_couplings(self) -> dict[tuple[int, int], float]:
        return dict(self._j_couplings)

    @property
    def local_fields(self) -> NDArray[np.float64] | None:
        return self._h_fields.copy() if self._h_fields is not None else None
