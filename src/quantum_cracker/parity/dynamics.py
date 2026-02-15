"""Parity dynamics simulator.

Implements three evolution modes for the parity Hamiltonian:
1. Exact unitary evolution (N <= 20, Schrodinger equation)
2. Glauber MCMC with PDQM-specific rates (pair hopping >> single hopping)
3. Simulated quantum annealing with parity-adaptive schedule

The PDQM-specific innovation: pair spin flips (parity-preserving) run at
rate t2 (unsuppressed), while single spin flips (parity-flipping) run at
rate t1 = t0 * exp(-Delta_E / kT) (exponentially suppressed at low T).

For large curves (N > 20), uses ECEnergyEvaluator for O(1) constraint
evaluation per spin flip via incremental point addition.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.types import (
    AnnealResult,
    AnnealSchedule,
    DynamicsSnapshot,
    ParityConfig,
)


def compute_parity(spins: NDArray[np.int8]) -> int:
    """Compute Z2 parity of a spin configuration.

    Product of all spins: +1 if even number of -1 spins, -1 if odd.
    """
    n_minus = int(np.sum(spins == -1))
    return 1 if n_minus % 2 == 0 else -1


class ParityDynamics:
    """Evolve a spin system under PDQM parity dynamics."""

    def __init__(
        self,
        hamiltonian: ParityHamiltonian,
        config: ParityConfig | None = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.config = config or hamiltonian.config
        self.history: list[DynamicsSnapshot] = []

    def _t1_effective(self, temperature: float) -> float:
        """t1(T) = t1_base * exp(-Delta_E / kT). Suppressed at low T."""
        if temperature <= 0:
            return 0.0
        return self.config.t1_base * np.exp(
            -self.config.delta_e / temperature
        )

    def _snapshot(
        self,
        step: int,
        spins: NDArray[np.int8],
        target_key: int | None = None,
    ) -> DynamicsSnapshot:
        """Create a dynamics snapshot without re-syncing the evaluator."""
        # Compute energy from Ising terms only (avoid re-syncing evaluator)
        e = 0.0
        if self.hamiltonian._h_fields is not None:
            e += float(np.dot(self.hamiltonian._h_fields, spins))
        for (i, j), j_val in self.hamiltonian._j_couplings.items():
            e += j_val * spins[i] * spins[j]
        # Constraint from evaluator (already in sync) or diagonal
        if self.hamiltonian._constraint_diagonal is not None:
            idx = ParityHamiltonian._spins_to_index(spins, len(spins))
            e += self.hamiltonian.config.constraint_weight * self.hamiltonian._constraint_diagonal[idx]
        elif self.hamiltonian._ec_evaluator is not None:
            e += self.hamiltonian.config.constraint_weight * self.hamiltonian._ec_evaluator.constraint_penalty()

        parity = compute_parity(spins)
        magnetization = float(np.mean(spins))
        overlap = None
        if target_key is not None:
            recovered = ParityHamiltonian.spins_to_key(spins)
            n = len(spins)
            xor = recovered ^ target_key
            matching = n - bin(xor).count("1")
            overlap = matching / n

        return DynamicsSnapshot(
            step=step,
            spins=spins.copy(),
            energy=e,
            parity=parity,
            magnetization=magnetization,
            overlap_with_target=overlap,
        )

    def evolve_exact(
        self,
        psi0: NDArray[np.complex128],
        t_final: float,
        dt: float,
    ) -> list[DynamicsSnapshot]:
        """Exact Schrodinger evolution via eigendecomposition (N <= 20)."""
        H = self.hamiltonian.to_matrix()
        eigenvalues, eigenvectors = scipy.linalg.eigh(H)
        coeffs = eigenvectors.T @ psi0

        snapshots = []
        n_steps = int(t_final / dt)
        n = self.hamiltonian.n_spins

        for step in range(n_steps + 1):
            t = step * dt
            phases = np.exp(-1j * eigenvalues * t)
            psi_t = eigenvectors @ (coeffs * phases)
            probs = np.abs(psi_t) ** 2
            max_idx = int(np.argmax(probs))
            spins = ParityHamiltonian._index_to_spins(max_idx, n)
            energy = float(eigenvalues @ (np.abs(coeffs) ** 2))

            snap = DynamicsSnapshot(
                step=step,
                spins=spins.copy(),
                energy=energy,
                parity=compute_parity(spins),
                magnetization=float(np.mean(spins)),
                overlap_with_target=None,
            )
            snapshots.append(snap)

        self.history = snapshots
        return snapshots

    def evolve_glauber(
        self,
        sigma0: NDArray[np.int8],
        n_sweeps: int,
        temperature: float | None = None,
        target_key: int | None = None,
        log_interval: int = 10,
        rng: np.random.Generator | None = None,
    ) -> list[DynamicsSnapshot]:
        """Glauber dynamics with PDQM-specific flip rates.

        Single-spin flips: acceptance *= t1/t2 (parity-flipping, suppressed)
        Pair-spin flips: standard Metropolis (parity-preserving, unsuppressed)
        """
        if rng is None:
            rng = np.random.default_rng()
        if temperature is None:
            temperature = self.config.temperature

        n = self.hamiltonian.n_spins
        spins = sigma0.copy()
        self.hamiltonian.sync_evaluator(spins)

        t1 = self._t1_effective(temperature)
        t2 = self.config.t2
        parity_suppression = t1 / t2 if t2 > 0 else 0.0
        beta = 1.0 / temperature if temperature > 0 else float("inf")

        snapshots = []
        snapshots.append(self._snapshot(0, spins, target_key))

        for sweep in range(1, n_sweeps + 1):
            # Single-spin updates (parity-flipping, suppressed)
            for _ in range(n):
                i = int(rng.integers(0, n))
                dE = self.hamiltonian.energy_change_single_flip(spins, i)

                if dE <= 0:
                    accept_prob = parity_suppression
                else:
                    accept_prob = parity_suppression * np.exp(-beta * dE)

                if rng.random() < accept_prob:
                    self.hamiltonian.apply_single_flip(spins, i)

            # Pair-spin updates (parity-preserving, unsuppressed)
            for _ in range(n):
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n))
                if i == j:
                    continue
                dE = self.hamiltonian.energy_change_pair_flip(spins, i, j)

                if dE <= 0:
                    accept_prob = 1.0
                else:
                    accept_prob = np.exp(-beta * dE)

                if rng.random() < accept_prob:
                    self.hamiltonian.apply_pair_flip(spins, i, j)

            if sweep % log_interval == 0 or sweep == n_sweeps:
                snapshots.append(self._snapshot(sweep, spins, target_key))

        self.history = snapshots
        return snapshots

    def evolve_standard_mcmc(
        self,
        sigma0: NDArray[np.int8],
        n_sweeps: int,
        temperature: float | None = None,
        target_key: int | None = None,
        log_interval: int = 10,
        rng: np.random.Generator | None = None,
    ) -> list[DynamicsSnapshot]:
        """Standard Metropolis MCMC (no parity weighting). Baseline."""
        if rng is None:
            rng = np.random.default_rng()
        if temperature is None:
            temperature = self.config.temperature

        n = self.hamiltonian.n_spins
        spins = sigma0.copy()
        self.hamiltonian.sync_evaluator(spins)

        beta = 1.0 / temperature if temperature > 0 else float("inf")

        snapshots = []
        snapshots.append(self._snapshot(0, spins, target_key))

        for sweep in range(1, n_sweeps + 1):
            for _ in range(n):
                i = int(rng.integers(0, n))
                dE = self.hamiltonian.energy_change_single_flip(spins, i)

                if dE <= 0:
                    self.hamiltonian.apply_single_flip(spins, i)
                elif rng.random() < np.exp(-beta * dE):
                    self.hamiltonian.apply_single_flip(spins, i)

            if sweep % log_interval == 0 or sweep == n_sweeps:
                snapshots.append(self._snapshot(sweep, spins, target_key))

        self.history = snapshots
        return snapshots

    def anneal(
        self,
        schedule: AnnealSchedule | None = None,
        n_reads: int = 10,
        target_key: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[AnnealResult]:
        """Simulated quantum annealing with parity-adaptive schedule."""
        if rng is None:
            rng = np.random.default_rng()
        if schedule is None:
            schedule = AnnealSchedule()

        n = self.hamiltonian.n_spins
        results = []

        for read in range(n_reads):
            spins = rng.choice([-1, 1], size=n).astype(np.int8)
            self.hamiltonian.sync_evaluator(spins)

            trajectory: list[DynamicsSnapshot] = []
            n_parity_flips = 0
            prev_parity = compute_parity(spins)

            for step in range(schedule.n_steps):
                frac = step / max(schedule.n_steps - 1, 1)

                if schedule.schedule_type == "linear":
                    beta = (
                        schedule.beta_initial
                        + frac * (schedule.beta_final - schedule.beta_initial)
                    )
                elif schedule.schedule_type == "exponential":
                    log_bi = np.log(max(schedule.beta_initial, 1e-10))
                    log_bf = np.log(max(schedule.beta_final, 1e-10))
                    beta = np.exp(log_bi + frac * (log_bf - log_bi))
                else:  # parity_adaptive
                    beta = (
                        schedule.beta_initial
                        + frac * (schedule.beta_final - schedule.beta_initial)
                    )
                    current_parity = compute_parity(spins)
                    if current_parity == 1:
                        beta *= 1.0 + self.config.delta_e * frac

                temperature = 1.0 / beta if beta > 0 else float("inf")
                t1 = self._t1_effective(temperature)
                t2 = self.config.t2
                parity_suppression = t1 / t2 if t2 > 0 else 0.0

                # Single-spin update (suppressed)
                i = int(rng.integers(0, n))
                dE = self.hamiltonian.energy_change_single_flip(spins, i)
                if dE <= 0:
                    accept = parity_suppression
                else:
                    accept = parity_suppression * np.exp(-beta * dE)
                if rng.random() < accept:
                    self.hamiltonian.apply_single_flip(spins, i)

                # Pair-spin update (unsuppressed)
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n))
                if i != j:
                    dE = self.hamiltonian.energy_change_pair_flip(spins, i, j)
                    if dE <= 0:
                        accept = 1.0
                    else:
                        accept = np.exp(-beta * dE)
                    if rng.random() < accept:
                        self.hamiltonian.apply_pair_flip(spins, i, j)

                new_parity = compute_parity(spins)
                if new_parity != prev_parity:
                    n_parity_flips += 1
                    prev_parity = new_parity

                if step % max(schedule.n_steps // 20, 1) == 0:
                    trajectory.append(
                        self._snapshot(step, spins, target_key)
                    )

            final_parity = compute_parity(spins)
            final_energy = self.hamiltonian.energy(spins)

            results.append(
                AnnealResult(
                    final_spins=spins.copy(),
                    final_energy=final_energy,
                    parity=final_parity,
                    n_parity_flips=n_parity_flips,
                    trajectory=trajectory,
                )
            )

        return results
