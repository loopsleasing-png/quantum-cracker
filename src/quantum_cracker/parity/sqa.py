"""Simulated Quantum Annealing via Suzuki-Trotter decomposition.

Maps the quantum partition function Z = Tr[exp(-beta * H)] with transverse
field Gamma to a (d+1)-dimensional classical system with P Trotter replicas
coupled along imaginary time.

H_eff = (1/P) * sum_r H_Ising(sigma^r)
      - J_perp * sum_r sum_i sigma_i^r * sigma_i^{r+1}

where J_perp = -(P*T/2) * ln(tanh(Gamma / (P*T)))

PDQM enhancement: even-parity replicas get enhanced J_perp (longer coherence),
odd-parity replicas get suppressed J_perp.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantum_cracker.parity.dynamics import compute_parity
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.types import (
    DynamicsSnapshot,
    ParityConfig,
    SQAResult,
    SQASchedule,
)


class SQAEngine:
    """Simulated Quantum Annealing via Suzuki-Trotter decomposition.

    Maintains P replicas of the spin system, each with its own
    ECEnergyEvaluator for O(1) constraint evaluation. Replicas are
    coupled along imaginary time by J_perp (transverse field).
    """

    def __init__(
        self,
        hamiltonian: ParityHamiltonian,
        config: ParityConfig | None = None,
    ) -> None:
        self.h = hamiltonian
        self.config = config or hamiltonian.config
        self.n_spins = hamiltonian.n_spins

        # Replica state (initialized in anneal())
        self.replica_spins: list[NDArray[np.int8]] = []
        self.replica_evaluators: list = []

    def _init_replicas(
        self, n_replicas: int, rng: np.random.Generator,
    ) -> None:
        """Initialize P replicas with random spins and independent evaluators."""
        n = self.n_spins
        self.replica_spins = [
            rng.choice([-1, 1], size=n).astype(np.int8)
            for _ in range(n_replicas)
        ]

        # Each replica gets its own evaluator copy
        if self.h._ec_evaluator is not None:
            self.replica_evaluators = []
            for r in range(n_replicas):
                ev = self.h._ec_evaluator.copy()
                ev.set_state_from_spins(self.replica_spins[r])
                self.replica_evaluators.append(ev)
        else:
            self.replica_evaluators = [None] * n_replicas

    @staticmethod
    def j_perp(gamma: float, temperature: float, n_replicas: int) -> float:
        """Inter-replica coupling strength from transverse field.

        J_perp = -(P*T/2) * ln(tanh(Gamma / (P*T)))

        Small Gamma (classical): J_perp -> inf (replicas frozen together)
        Large Gamma (quantum): J_perp -> 0 (replicas independent)
        """
        if temperature <= 0:
            return 1e10
        if gamma <= 0:
            return 1e10  # no transverse field -> replicas frozen
        pt = n_replicas * temperature
        arg = gamma / pt
        if arg > 15.0:
            # Asymptotic: J_perp ≈ P*T*exp(-2*Gamma/(P*T)) for large arg
            return pt * np.exp(-2.0 * arg)
        tanh_val = np.tanh(arg)
        if tanh_val <= 0:
            return 1e10
        return -(pt / 2.0) * np.log(tanh_val)

    def _j_perp_parity(
        self, j_perp_base: float, parity: int, delta_e: float,
        temperature: float, n_replicas: int,
    ) -> float:
        """PDQM parity-weighted J_perp.

        Even parity: J_perp * exp(+delta_e / (2*P*T))  -- enhanced coherence
        Odd parity:  J_perp * exp(-delta_e / (2*P*T))  -- suppressed coherence
        """
        if temperature <= 0 or n_replicas <= 0:
            return j_perp_base
        exponent = delta_e / (2.0 * n_replicas * temperature)
        if parity == 1:  # even parity
            return j_perp_base * np.exp(exponent)
        else:  # odd parity
            return j_perp_base * np.exp(-exponent)

    def _intra_de_single(
        self, replica_idx: int, spin_idx: int,
    ) -> float:
        """Intra-replica energy change for flipping one spin.

        Same as ParityHamiltonian.energy_change_single_flip but uses
        the replica's own evaluator.
        """
        spins = self.replica_spins[replica_idx]
        s_i = spins[spin_idx]
        dE = 0.0

        if self.h._h_fields is not None:
            dE += -2.0 * self.h._h_fields[spin_idx] * s_i

        for (i, j), j_val in self.h._j_couplings.items():
            if i == spin_idx:
                dE += -2.0 * j_val * s_i * spins[j]
            elif j == spin_idx:
                dE += -2.0 * j_val * spins[i] * s_i

        # Constraint term
        if self.h._constraint_diagonal is not None:
            n = self.n_spins
            idx_before = ParityHamiltonian._spins_to_index(spins, n)
            flipped = spins.copy()
            flipped[spin_idx] *= -1
            idx_after = ParityHamiltonian._spins_to_index(flipped, n)
            dE += self.h.config.constraint_weight * (
                self.h._constraint_diagonal[idx_after]
                - self.h._constraint_diagonal[idx_before]
            )
        elif self.replica_evaluators[replica_idx] is not None:
            ev = self.replica_evaluators[replica_idx]
            old_pen = ev.constraint_penalty()
            new_pen = ev.peek_flip_single(spin_idx)
            dE += self.h.config.constraint_weight * (new_pen - old_pen)

        return dE

    def _intra_de_pair(
        self, replica_idx: int, spin_a: int, spin_b: int,
    ) -> float:
        """Intra-replica energy change for parity-preserving pair flip."""
        spins = self.replica_spins[replica_idx]
        s_a = spins[spin_a]
        s_b = spins[spin_b]
        dE = 0.0

        if self.h._h_fields is not None:
            dE += -2.0 * self.h._h_fields[spin_a] * s_a
            dE += -2.0 * self.h._h_fields[spin_b] * s_b

        for (i, j), j_val in self.h._j_couplings.items():
            if i == spin_a and j == spin_b:
                pass  # both flip: product unchanged
            elif i == spin_a:
                dE += -2.0 * j_val * s_a * spins[j]
            elif j == spin_a:
                dE += -2.0 * j_val * spins[i] * s_a
            elif i == spin_b:
                dE += -2.0 * j_val * s_b * spins[j]
            elif j == spin_b:
                dE += -2.0 * j_val * spins[i] * s_b

        if self.h._constraint_diagonal is not None:
            n = self.n_spins
            idx_before = ParityHamiltonian._spins_to_index(spins, n)
            flipped = spins.copy()
            flipped[spin_a] *= -1
            flipped[spin_b] *= -1
            idx_after = ParityHamiltonian._spins_to_index(flipped, n)
            dE += self.h.config.constraint_weight * (
                self.h._constraint_diagonal[idx_after]
                - self.h._constraint_diagonal[idx_before]
            )
        elif self.replica_evaluators[replica_idx] is not None:
            ev = self.replica_evaluators[replica_idx]
            old_pen = ev.constraint_penalty()
            new_pen = ev.peek_flip_pair(spin_a, spin_b)
            dE += self.h.config.constraint_weight * (new_pen - old_pen)

        return dE

    def _apply_single_flip(self, replica_idx: int, spin_idx: int) -> None:
        """Flip one spin in a replica and update its evaluator."""
        self.replica_spins[replica_idx][spin_idx] *= -1
        if self.replica_evaluators[replica_idx] is not None:
            self.replica_evaluators[replica_idx].flip_single(spin_idx)

    def _apply_pair_flip(
        self, replica_idx: int, spin_a: int, spin_b: int,
    ) -> None:
        """Flip two spins in a replica and update its evaluator."""
        self.replica_spins[replica_idx][spin_a] *= -1
        self.replica_spins[replica_idx][spin_b] *= -1
        if self.replica_evaluators[replica_idx] is not None:
            self.replica_evaluators[replica_idx].flip_single(spin_a)
            self.replica_evaluators[replica_idx].flip_single(spin_b)

    def _replica_energy(self, replica_idx: int) -> float:
        """Full energy of one replica (Ising + constraint)."""
        spins = self.replica_spins[replica_idx]
        e = 0.0

        if self.h._h_fields is not None:
            e += float(np.dot(self.h._h_fields, spins))

        for (i, j), j_val in self.h._j_couplings.items():
            e += j_val * spins[i] * spins[j]

        if self.h._constraint_diagonal is not None:
            idx = ParityHamiltonian._spins_to_index(spins, self.n_spins)
            e += self.h.config.constraint_weight * self.h._constraint_diagonal[idx]
        elif self.replica_evaluators[replica_idx] is not None:
            e += self.h.config.constraint_weight * self.replica_evaluators[replica_idx].constraint_penalty()

        return e

    def _try_build_c_model(self) -> object | None:
        """Build a CIsingModel from the current Hamiltonian, or None."""
        try:
            from quantum_cracker.accel._ising import CIsingModel
            return CIsingModel(
                n_spins=self.n_spins,
                h_fields=self.h._h_fields,
                j_couplings=self.h._j_couplings,
                constraint_weight=self.h.config.constraint_weight,
                constraint_diagonal=self.h._constraint_diagonal,
            )
        except (ImportError, RuntimeError):
            return None

    def _try_build_c_evaluators(
        self, n_replicas: int, rng: np.random.Generator,
    ) -> list | None:
        """Build C EC evaluators for each replica, or None."""
        try:
            from quantum_cracker.accel._ec_arith import CECEvaluator
        except (ImportError, RuntimeError):
            return None

        if not self.replica_evaluators or self.replica_evaluators[0] is None:
            return None

        # Check if evaluators are already CECEvaluators
        if isinstance(self.replica_evaluators[0], CECEvaluator):
            return self.replica_evaluators

        return None

    def anneal(
        self,
        schedule: SQASchedule | None = None,
        n_reads: int = 1,
        target_key: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> SQAResult:
        """Run Simulated Quantum Annealing.

        Uses C-accelerated sweep kernel when available, falling back
        to pure Python otherwise.

        Args:
            schedule: SQA annealing schedule (gamma, beta, replicas)
            n_reads: Number of independent annealing runs (best-of)
            target_key: True key for bit match tracking
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()
        if schedule is None:
            schedule = SQASchedule()

        P = schedule.n_replicas
        n = self.n_spins
        delta_e = self.config.delta_e
        parity_weighted = schedule.parity_weighted

        # Parity suppression for single-spin flips
        t1_base = self.config.t1_base
        t2 = self.config.t2

        # Try to build C-accelerated model for sweep kernel
        c_model = self._try_build_c_model()
        use_c_sweep = c_model is not None
        sqa_sweep_c = None
        if use_c_sweep:
            try:
                from quantum_cracker.accel._ising import sqa_sweep_c as _sqa_sweep_c
                sqa_sweep_c = _sqa_sweep_c
            except (ImportError, RuntimeError):
                use_c_sweep = False

        best_overall_energy = float("inf")
        best_overall_spins: NDArray[np.int8] | None = None
        best_overall_replica = 0
        total_parity_flips = 0
        trajectory: list[DynamicsSnapshot] = []

        for _read in range(n_reads):
            self._init_replicas(P, rng)

            # Track parity per replica
            replica_parities = [
                compute_parity(self.replica_spins[r]) for r in range(P)
            ]

            for step in range(schedule.n_steps):
                frac = step / max(schedule.n_steps - 1, 1)

                # Annealing schedule: linear interpolation
                gamma = schedule.gamma_initial + frac * (
                    schedule.gamma_final - schedule.gamma_initial
                )
                beta = schedule.beta_initial + frac * (
                    schedule.beta_final - schedule.beta_initial
                )
                temperature = 1.0 / beta if beta > 0 else float("inf")

                # Compute base J_perp
                jp_base = self.j_perp(gamma, temperature, P)

                # Parity-suppressed single-flip rate
                if temperature > 0:
                    parity_suppression = t1_base * np.exp(-delta_e / temperature) / t2
                else:
                    parity_suppression = 0.0

                # Use C sweep kernel if available
                if use_c_sweep and sqa_sweep_c is not None:
                    # Build C evaluator list from replica_evaluators
                    c_ec_evs = self.replica_evaluators if (
                        self.replica_evaluators and
                        self.replica_evaluators[0] is not None
                    ) else [None] * P

                    sqa_sweep_c(
                        c_model,
                        self.replica_spins,
                        c_ec_evs,
                        jp_base,
                        parity_suppression,
                        beta,
                        delta_e,
                        temperature,
                        parity_weighted,
                        rng,
                    )

                    # Update parity tracking
                    for r in range(P):
                        replica_parities[r] = compute_parity(self.replica_spins[r])
                else:
                    # Pure Python sweep (original implementation)
                    self._python_sweep(
                        P, n, parity_weighted, jp_base, delta_e,
                        temperature, parity_suppression, beta,
                        replica_parities, total_parity_flips, rng,
                    )
                    # Update parity tracking
                    for r in range(P):
                        new_p = compute_parity(self.replica_spins[r])
                        if new_p != replica_parities[r]:
                            total_parity_flips += 1
                        replica_parities[r] = new_p

                # Log trajectory periodically
                if step % max(schedule.n_steps // 20, 1) == 0:
                    best_r = min(range(P), key=lambda r: self._replica_energy(r))
                    best_spins = self.replica_spins[best_r]
                    overlap = None
                    if target_key is not None:
                        ek = ParityHamiltonian.spins_to_key(best_spins)
                        xor = ek ^ target_key
                        overlap = (n - bin(xor).count("1")) / n

                    trajectory.append(DynamicsSnapshot(
                        step=step,
                        spins=best_spins.copy(),
                        energy=self._replica_energy(best_r),
                        parity=compute_parity(best_spins),
                        magnetization=float(np.mean(best_spins)),
                        overlap_with_target=overlap,
                    ))

            # Find best replica from this read
            for r in range(P):
                e_r = self._replica_energy(r)
                if e_r < best_overall_energy:
                    best_overall_energy = e_r
                    best_overall_spins = self.replica_spins[r].copy()
                    best_overall_replica = r

        # Collect final state
        assert best_overall_spins is not None
        final_replicas = [s.copy() for s in self.replica_spins]
        final_energies = [self._replica_energy(r) for r in range(P)]

        # Extract key via majority vote across replicas
        extracted_key = self._majority_vote_key(final_replicas)

        # Compute bit match if target known
        bit_match = None
        if target_key is not None:
            xor = extracted_key ^ target_key
            bit_match = (n - bin(xor).count("1")) / n

        # Replica agreement: fraction of bits where all replicas agree
        agreement = self._replica_agreement(final_replicas)

        return SQAResult(
            final_replicas=final_replicas,
            final_energies=final_energies,
            best_spins=best_overall_spins,
            best_energy=best_overall_energy,
            best_replica=best_overall_replica,
            extracted_key=extracted_key,
            bit_match_rate=bit_match,
            replica_agreement=agreement,
            n_parity_flips=total_parity_flips,
            trajectory=trajectory,
        )

    def _python_sweep(
        self,
        P: int, n: int, parity_weighted: bool,
        jp_base: float, delta_e: float, temperature: float,
        parity_suppression: float, beta: float,
        replica_parities: list[int],
        total_parity_flips: int,
        rng: np.random.Generator,
    ) -> None:
        """Run one SQA sweep step in pure Python (fallback)."""
        for r in range(P):
            spins_r = self.replica_spins[r]
            r_prev = (r - 1) % P
            r_next = (r + 1) % P

            # Per-replica J_perp (parity-weighted or uniform)
            if parity_weighted:
                jp_r = self._j_perp_parity(
                    jp_base, replica_parities[r], delta_e,
                    temperature, P,
                )
            else:
                jp_r = jp_base

            # Single-spin flip proposals (N proposals per replica)
            for _ in range(n):
                i = int(rng.integers(0, n))

                # Intra-replica dE scaled by 1/P
                dE_intra = self._intra_de_single(r, i) / P

                # Inter-replica coupling dE
                s_i = spins_r[i]
                s_prev = self.replica_spins[r_prev][i]
                s_next = self.replica_spins[r_next][i]
                dE_inter = jp_r * (-2.0) * s_i * (s_prev + s_next)

                dE = dE_intra + dE_inter

                # Accept with parity suppression (single flips change parity)
                if dE <= 0:
                    accept_prob = parity_suppression
                else:
                    accept_prob = parity_suppression * np.exp(-beta * dE)

                if rng.random() < accept_prob:
                    self._apply_single_flip(r, i)
                    replica_parities[r] *= -1

            # Pair-spin flip proposals (N proposals per replica)
            for _ in range(n):
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n))
                if i == j:
                    continue

                dE_intra = self._intra_de_pair(r, i, j) / P

                s_i = spins_r[i]
                s_j = spins_r[j]
                s_prev_i = self.replica_spins[r_prev][i]
                s_next_i = self.replica_spins[r_next][i]
                s_prev_j = self.replica_spins[r_prev][j]
                s_next_j = self.replica_spins[r_next][j]
                dE_inter = jp_r * (-2.0) * (
                    s_i * (s_prev_i + s_next_i)
                    + s_j * (s_prev_j + s_next_j)
                )

                dE = dE_intra + dE_inter

                # Pair flips preserve parity -- unsuppressed
                if dE <= 0:
                    accept_prob = 1.0
                else:
                    accept_prob = np.exp(-beta * dE)

                if rng.random() < accept_prob:
                    self._apply_pair_flip(r, i, j)

    def _majority_vote_key(self, replicas: list[NDArray[np.int8]]) -> int:
        """Extract key via per-bit majority vote across all replicas."""
        n = self.n_spins
        P = len(replicas)
        key = 0
        for bit in range(n):
            vote = sum(1 for r in range(P) if replicas[r][bit] == -1)
            if vote > P // 2:
                key |= 1 << bit
        return key

    def _replica_agreement(self, replicas: list[NDArray[np.int8]]) -> float:
        """Fraction of bits where all replicas have the same value."""
        n = self.n_spins
        P = len(replicas)
        if P <= 1:
            return 1.0
        agree = 0
        for bit in range(n):
            vals = {replicas[r][bit] for r in range(P)}
            if len(vals) == 1:
                agree += 1
        return agree / n
