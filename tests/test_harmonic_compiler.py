"""Tests for the harmonic compiler."""

import numpy as np
import pytest

from quantum_cracker.core.harmonic_compiler import HarmonicCompiler
from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.utils.types import SimulationConfig


@pytest.fixture
def initialized_grid():
    """A 20x20x20 grid initialized from a random key (small for speed)."""
    grid = SphericalVoxelGrid(size=20)
    key = KeyInput(42)
    grid.initialize_from_key(key)
    return grid


@pytest.fixture
def compiler(initialized_grid):
    return HarmonicCompiler(initialized_grid)


class TestVibrationField:
    def test_shape(self, compiler):
        field = compiler._build_vibration_field(0.0)
        assert field.shape == (20, 20, 20)

    def test_bounded(self, compiler):
        field = compiler._build_vibration_field(0.5)
        # sin * cos is bounded in [-1, 1]
        assert np.abs(field).max() <= 1.0 + 1e-10

    def test_varies_with_time(self, compiler):
        f1 = compiler._build_vibration_field(0.0)
        f2 = compiler._build_vibration_field(1.0)
        assert not np.allclose(f1, f2)


class TestApplyResonance:
    def test_modifies_amplitude(self, compiler, initialized_grid):
        amp_before = initialized_grid.amplitude.copy()
        compiler.apply_resonance(t=0.5)
        assert not np.allclose(initialized_grid.amplitude, amp_before)

    def test_updates_energy(self, compiler, initialized_grid):
        compiler.apply_resonance(t=0.5)
        np.testing.assert_allclose(
            initialized_grid.energy,
            np.abs(initialized_grid.amplitude) ** 2,
        )

    def test_sets_time(self, compiler):
        compiler.apply_resonance(t=2.5)
        assert compiler.time == 2.5


class TestSHFilter:
    def test_filter_modifies_grid(self, initialized_grid):
        amp_before = initialized_grid.amplitude.copy()
        compiler = HarmonicCompiler(initialized_grid)
        compiler.apply_spherical_harmonic_filter(l_target=5)
        # Should be different after filtering
        assert not np.allclose(initialized_grid.amplitude, amp_before)

    def test_filter_updates_energy(self, initialized_grid):
        compiler = HarmonicCompiler(initialized_grid)
        compiler.apply_spherical_harmonic_filter(l_target=5)
        np.testing.assert_allclose(
            initialized_grid.energy,
            np.abs(initialized_grid.amplitude) ** 2,
        )


class TestHamiltonian:
    def test_eigenvalues_shape(self, compiler):
        eigs = compiler.compute_hamiltonian_eigenvalues(shell_index=-1)
        # For a 20x20 angular grid, matrix is 400x400
        assert len(eigs) == 400

    def test_eigenvalues_sorted(self, compiler):
        eigs = compiler.compute_hamiltonian_eigenvalues()
        assert np.all(np.diff(eigs) >= -1e-10)  # sorted ascending

    def test_eigenvalues_real(self, compiler):
        eigs = compiler.compute_hamiltonian_eigenvalues()
        assert np.all(np.isreal(eigs))


class TestPeakExtraction:
    def test_extracts_peaks(self, compiler):
        peaks = compiler.extract_peaks(num_peaks=20)
        assert len(peaks) > 0
        assert len(peaks) <= 20

    def test_peaks_sorted_by_energy(self, compiler):
        peaks = compiler.extract_peaks(num_peaks=20)
        if len(peaks) > 1:
            energies = [p.energy for p in peaks]
            assert all(energies[i] >= energies[i + 1] for i in range(len(energies) - 1))

    def test_peaks_have_correct_fields(self, compiler):
        peaks = compiler.extract_peaks(num_peaks=5)
        if peaks:
            p = peaks[0]
            assert isinstance(p.grid_index, tuple)
            assert len(p.grid_index) == 3
            assert isinstance(p.r, float)
            assert isinstance(p.theta, float)
            assert isinstance(p.phi, float)
            assert isinstance(p.amplitude, float)
            assert isinstance(p.energy, float)

    def test_zero_grid_no_peaks(self):
        grid = SphericalVoxelGrid(size=10)
        compiler = HarmonicCompiler(grid)
        peaks = compiler.extract_peaks()
        assert len(peaks) == 0


class TestCompilePipeline:
    def test_compile_returns_peaks(self, initialized_grid):
        compiler = HarmonicCompiler(initialized_grid)
        peaks = compiler.compile(num_steps=10, dt=0.01)
        assert isinstance(peaks, list)
        assert len(peaks) > 0

    def test_compile_advances_time(self, initialized_grid):
        compiler = HarmonicCompiler(initialized_grid)
        compiler.compile(num_steps=10, dt=0.1)
        assert compiler.time == pytest.approx(1.0, rel=1e-10)

    def test_compile_with_sh_filter(self, initialized_grid):
        compiler = HarmonicCompiler(initialized_grid)
        peaks = compiler.compile(
            num_steps=10,
            dt=0.01,
            apply_sh_filter=True,
            sh_filter_interval=5,
        )
        assert isinstance(peaks, list)

    def test_different_keys_different_amplitudes(self):
        g1 = SphericalVoxelGrid(size=15)
        g1.initialize_from_key(KeyInput(0))
        c1 = HarmonicCompiler(g1)
        c1.compile(num_steps=5)

        g2 = SphericalVoxelGrid(size=15)
        g2.initialize_from_key(KeyInput(2**256 - 1))
        c2 = HarmonicCompiler(g2)
        c2.compile(num_steps=5)

        # The final grid amplitudes should differ for different keys
        assert not np.allclose(g1.amplitude, g2.amplitude)
