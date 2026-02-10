"""Tests for the matplotlib plot suite."""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.utils.types import Peak
from quantum_cracker.visualization.plots import PlotSuite


@pytest.fixture
def tmp_save_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def plot_suite(tmp_save_dir):
    return PlotSuite(save_dir=tmp_save_dir)


@pytest.fixture
def small_grid():
    grid = SphericalVoxelGrid(size=15)
    grid.initialize_from_key(KeyInput(42))
    return grid


@pytest.fixture
def sample_peaks():
    return [
        Peak(
            grid_index=(i, i, i),
            r=0.5,
            theta=np.pi * i / 10,
            phi=2 * np.pi * i / 10,
            amplitude=float(10 - i) / 10,
            energy=((10 - i) / 10) ** 2,
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_history():
    return [
        {
            "tick": i + 1,
            "radius": 1e-35 * 1.01 ** (i + 1),
            "avg_gap": 0.1,
            "min_gap": 0.05,
            "max_gap": 0.15,
            "num_visible": min(i * 5, 256),
            "all_visible": False,
            "min_physical_gap": 5e-37,
            "max_physical_gap": 1.5e-36,
        }
        for i in range(50)
    ]


class TestSHHeatmap:
    def test_returns_figure(self, plot_suite, small_grid):
        fig = plot_suite.spherical_harmonic_heatmap(small_grid, save=False)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, plot_suite, small_grid, tmp_save_dir):
        plot_suite.spherical_harmonic_heatmap(small_grid, save=True)
        assert os.path.exists(os.path.join(tmp_save_dir, "qc_sh_heatmap.png"))


class TestGapVsTime:
    def test_returns_figure(self, plot_suite, sample_history):
        fig = plot_suite.thread_gap_vs_time(sample_history, save=False)
        assert isinstance(fig, plt.Figure)

    def test_empty_history(self, plot_suite):
        fig = plot_suite.thread_gap_vs_time([], save=False)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, plot_suite, sample_history, tmp_save_dir):
        plot_suite.thread_gap_vs_time(sample_history, save=True)
        assert os.path.exists(os.path.join(tmp_save_dir, "qc_gap_vs_time.png"))


class TestEnergyLandscape:
    def test_returns_figure(self, plot_suite):
        eigs = np.sort(np.random.randn(100))
        fig = plot_suite.energy_landscape(eigs, save=False)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, plot_suite, tmp_save_dir):
        eigs = np.sort(np.random.randn(100))
        plot_suite.energy_landscape(eigs, save=True)
        assert os.path.exists(os.path.join(tmp_save_dir, "qc_energy_landscape.png"))


class TestKeyComparison:
    def test_returns_figure(self, plot_suite):
        key = KeyInput(42)
        fig = plot_suite.key_comparison(key, key.as_bits, save=False)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, plot_suite, tmp_save_dir):
        key = KeyInput(42)
        plot_suite.key_comparison(key, [0] * 256, save=True)
        assert os.path.exists(os.path.join(tmp_save_dir, "qc_key_comparison.png"))


class TestPeak3D:
    def test_returns_figure(self, plot_suite, sample_peaks):
        fig = plot_suite.peak_distribution_3d(sample_peaks, save=False)
        assert isinstance(fig, plt.Figure)

    def test_empty_peaks(self, plot_suite):
        fig = plot_suite.peak_distribution_3d([], save=False)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, plot_suite, sample_peaks, tmp_save_dir):
        plot_suite.peak_distribution_3d(sample_peaks, save=True)
        assert os.path.exists(os.path.join(tmp_save_dir, "qc_peak_3d.png"))
