"""Tests for C accelerator layer.

Verifies that the C implementations produce identical results to
the pure Python implementations for EC arithmetic, Ising energy
evaluation, and SQA sweeps.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cracker.accel import is_available
from quantum_cracker.parity.ec_constraints import (
    ECConstraintEncoder,
    ECEnergyEvaluator,
    SmallEC,
    make_curve,
)
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.types import ParityConfig

# Skip all tests if C library not available
pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="C accelerator library not built",
)


class TestCECEvaluator:
    """Test C EC evaluator against Python reference."""

    def setup_method(self) -> None:
        """Set up a small test curve."""
        self.curve = SmallEC(97, 0, 7)
        G = self.curve.generator
        # Pick private key = 5
        self.key = 5
        P = self.curve.multiply(G, self.key)
        assert P is not None
        self.pub = P
        self.n_bits = self.curve.key_bit_length()

        # Python reference evaluator
        self.py_ev = ECEnergyEvaluator(
            self.curve, G, self.pub, self.n_bits,
        )

        # C evaluator (via make_evaluator)
        encoder = ECConstraintEncoder(self.curve, G, self.pub)
        self.c_ev = encoder.make_evaluator()

    def test_set_state_correct_key(self) -> None:
        """C evaluator returns 0 penalty for correct key."""
        self.c_ev.set_state(self.key)
        assert self.c_ev.constraint_penalty() == 0.0

    def test_set_state_wrong_key(self) -> None:
        """C evaluator returns 1 penalty for wrong key."""
        self.c_ev.set_state(self.key + 1)
        assert self.c_ev.constraint_penalty() == 1.0

    def test_set_state_from_spins(self) -> None:
        """C evaluator handles spin-based state setting."""
        spins = ParityHamiltonian.key_to_spins(self.key, self.n_bits)
        self.c_ev.set_state_from_spins(spins)
        assert self.c_ev.constraint_penalty() == 0.0

        # Wrong key
        wrong_spins = ParityHamiltonian.key_to_spins(self.key + 1, self.n_bits)
        self.c_ev.set_state_from_spins(wrong_spins)
        assert self.c_ev.constraint_penalty() == 1.0

    def test_flip_single_matches_python(self) -> None:
        """flip_single produces same results as Python."""
        rng = np.random.default_rng(42)
        for trial_key in range(min(20, self.curve.order)):
            self.py_ev.set_state(trial_key)
            self.c_ev.set_state(trial_key)

            for bit in range(self.n_bits):
                py_pen = self.py_ev.peek_flip_single(bit)
                c_pen = self.c_ev.peek_flip_single(bit)
                assert py_pen == c_pen, (
                    f"Mismatch at key={trial_key}, bit={bit}: "
                    f"py={py_pen}, c={c_pen}"
                )

    def test_flip_pair_matches_python(self) -> None:
        """peek_flip_pair produces same results as Python."""
        for trial_key in [0, 1, 5, 10, 20]:
            if trial_key >= self.curve.order:
                continue
            self.py_ev.set_state(trial_key)
            self.c_ev.set_state(trial_key)

            for a in range(min(4, self.n_bits)):
                for b in range(a + 1, min(4, self.n_bits)):
                    py_pen = self.py_ev.peek_flip_pair(a, b)
                    c_pen = self.c_ev.peek_flip_pair(a, b)
                    assert py_pen == c_pen, (
                        f"Mismatch at key={trial_key}, bits=({a},{b})"
                    )

    def test_flip_single_mutates_state(self) -> None:
        """flip_single correctly updates internal state."""
        self.c_ev.set_state(5)
        # Flip bit 0: key 5 -> 4
        self.c_ev.flip_single(0)
        # Now verify by checking what key 4 would give
        self.py_ev.set_state(4)
        assert self.c_ev.constraint_penalty() == self.py_ev.constraint_penalty()

    def test_copy_independence(self) -> None:
        """Copied evaluator is independent from original."""
        self.c_ev.set_state(5)
        ev2 = self.c_ev.copy()

        # Mutate copy
        ev2.flip_single(0)

        # Original unchanged
        assert self.c_ev.constraint_penalty() == 0.0
        # Copy changed
        self.py_ev.set_state(4)
        assert ev2.constraint_penalty() == self.py_ev.constraint_penalty()


class TestCIsingModel:
    """Test C Ising model against Python reference."""

    def setup_method(self) -> None:
        """Build a small Hamiltonian for testing."""
        self.curve = SmallEC(97, 0, 7)
        G = self.curve.generator
        self.key = 5
        P = self.curve.multiply(G, self.key)
        assert P is not None

        config = ParityConfig(n_spins=self.curve.key_bit_length())
        self.h = ParityHamiltonian.from_ec_dlp(
            self.curve, G, P, config=config,
        )

    def test_energy_matches_python(self) -> None:
        """C Ising energy matches Python for random spin configs."""
        from quantum_cracker.accel._ising import CIsingModel

        c_model = CIsingModel(
            n_spins=self.h.n_spins,
            h_fields=self.h._h_fields,
            j_couplings=self.h._j_couplings,
            constraint_weight=self.h.config.constraint_weight,
            constraint_diagonal=self.h._constraint_diagonal,
        )

        rng = np.random.default_rng(42)
        for _ in range(20):
            spins = rng.choice([-1, 1], size=self.h.n_spins).astype(np.int8)
            py_e = self.h.energy(spins)
            c_e = c_model.energy(spins)
            assert abs(py_e - c_e) < 1e-10, f"Energy mismatch: {py_e} vs {c_e}"

    def test_delta_e_single_matches_python(self) -> None:
        """C delta-E single flip matches Python."""
        from quantum_cracker.accel._ising import CIsingModel

        c_model = CIsingModel(
            n_spins=self.h.n_spins,
            h_fields=self.h._h_fields,
            j_couplings=self.h._j_couplings,
            constraint_weight=self.h.config.constraint_weight,
            constraint_diagonal=self.h._constraint_diagonal,
        )

        rng = np.random.default_rng(123)
        for _ in range(10):
            spins = rng.choice([-1, 1], size=self.h.n_spins).astype(np.int8)
            self.h.sync_evaluator(spins)
            for flip in range(self.h.n_spins):
                py_de = self.h.energy_change_single_flip(spins, flip)
                c_de = c_model.delta_e_single(spins, flip)
                assert abs(py_de - c_de) < 1e-10, (
                    f"Delta-E mismatch at flip={flip}: {py_de} vs {c_de}"
                )

    def test_delta_e_pair_matches_python(self) -> None:
        """C delta-E pair flip matches Python."""
        from quantum_cracker.accel._ising import CIsingModel

        c_model = CIsingModel(
            n_spins=self.h.n_spins,
            h_fields=self.h._h_fields,
            j_couplings=self.h._j_couplings,
            constraint_weight=self.h.config.constraint_weight,
            constraint_diagonal=self.h._constraint_diagonal,
        )

        rng = np.random.default_rng(456)
        spins = rng.choice([-1, 1], size=self.h.n_spins).astype(np.int8)
        self.h.sync_evaluator(spins)
        for a in range(min(5, self.h.n_spins)):
            for b in range(a + 1, min(5, self.h.n_spins)):
                py_de = self.h.energy_change_pair_flip(spins, a, b)
                c_de = c_model.delta_e_pair(spins, a, b)
                assert abs(py_de - c_de) < 1e-10, (
                    f"Delta-E pair mismatch at ({a},{b}): {py_de} vs {c_de}"
                )


class TestHamiltonianIntegration:
    """Test that Hamiltonian correctly uses C evaluator when available."""

    def test_from_ec_dlp_ground_state(self) -> None:
        """Hamiltonian ground state matches known key with C evaluator."""
        curve = SmallEC(97, 0, 7)
        G = curve.generator
        key = 5
        P = curve.multiply(G, key)
        assert P is not None

        h = ParityHamiltonian.from_ec_dlp(
            curve, G, P,
            config=ParityConfig(n_spins=curve.key_bit_length()),
        )

        # Ground state should recover the key
        gs_key = h.ground_state_key()
        assert gs_key == key, f"Ground state key {gs_key} != expected {key}"

    def test_from_known_key_ground_state(self) -> None:
        """from_known_key produces correct ground state."""
        key_bits = [1, 0, 1, 1, 0]
        h = ParityHamiltonian.from_known_key(key_bits)
        gs_key = h.ground_state_key()
        expected = sum(b << i for i, b in enumerate(key_bits))
        assert gs_key == expected


class TestCAccelAvailability:
    """Test the availability detection mechanism."""

    def test_is_available(self) -> None:
        """C library should be available in test environment."""
        assert is_available()

    def test_make_evaluator_returns_c_type(self) -> None:
        """make_evaluator should return CECEvaluator when lib available."""
        from quantum_cracker.accel._ec_arith import CECEvaluator

        curve = SmallEC(97, 0, 7)
        G = curve.generator
        P = curve.multiply(G, 5)
        assert P is not None
        encoder = ECConstraintEncoder(curve, G, P)
        ev = encoder.make_evaluator()
        assert isinstance(ev, CECEvaluator)
