"""Tests for metric extraction."""

import numpy as np
import pytest

from quantum_cracker.analysis.metrics import MetricExtractor
from quantum_cracker.utils.types import Peak


def make_peaks(n: int = 10) -> list[Peak]:
    """Create n test peaks with varying properties."""
    return [
        Peak(
            grid_index=(i, i, i),
            r=0.5,
            theta=np.pi * i / n,
            phi=2 * np.pi * i / n,
            amplitude=float(n - i) / n,
            energy=((n - i) / n) ** 2,
        )
        for i in range(n)
    ]


def make_history(n: int = 10) -> list[dict]:
    """Create n history entries."""
    return [
        {
            "tick": i + 1,
            "radius": 1e-35 * (1.01 ** (i + 1)),
            "avg_gap": 0.1,
            "min_gap": 0.05,
            "max_gap": 0.15,
            "num_visible": min(i, 256),
            "all_visible": i >= 50,
            "min_physical_gap": 0.05 * 1e-35 * (1.01 ** (i + 1)),
            "max_physical_gap": 0.15 * 1e-35 * (1.01 ** (i + 1)),
        }
        for i in range(n)
    ]


class TestResonancePeakStats:
    def test_with_peaks(self):
        ext = MetricExtractor(make_peaks(10), [])
        stats = ext.resonance_peak_stats()
        assert stats["count"] == 10
        assert stats["amplitude_mean"] > 0
        assert stats["energy_max"] > 0

    def test_empty_peaks(self):
        ext = MetricExtractor([], [])
        stats = ext.resonance_peak_stats()
        assert stats["count"] == 0
        assert stats["amplitude_mean"] == 0.0


class TestThreadSeparationStats:
    def test_with_history(self):
        ext = MetricExtractor([], make_history(100))
        stats = ext.thread_separation_stats()
        assert stats["total_steps"] == 100
        assert stats["final_num_visible"] > 0

    def test_empty_history(self):
        ext = MetricExtractor([], [])
        stats = ext.thread_separation_stats()
        assert stats["total_steps"] == 0
        assert stats["first_visible_tick"] is None

    def test_visibility_timeline(self):
        history = make_history(100)
        # First visible at tick where num_visible > 0 (tick 2)
        ext = MetricExtractor([], history)
        stats = ext.thread_separation_stats()
        assert stats["first_visible_tick"] is not None


class TestEnergyLandscape:
    def test_with_eigenvalues(self):
        eigs = np.array([-5.0, -3.0, -1.0, 2.0, 4.0])
        ext = MetricExtractor([], [])
        stats = ext.energy_landscape_stats(eigs)
        assert stats["ground_state_energy"] == -5.0
        assert stats["energy_gap"] == 2.0
        assert stats["eigenvalue_count"] == 5

    def test_empty(self):
        ext = MetricExtractor([], [])
        stats = ext.energy_landscape_stats()
        assert stats["ground_state_energy"] is None


class TestPeaksToKeyBits:
    def test_returns_256_bits(self):
        ext = MetricExtractor(make_peaks(78), [])
        bits = ext.peaks_to_key_bits()
        assert len(bits) == 256
        assert all(b in (0, 1) for b in bits)

    def test_empty_peaks_returns_zeros(self):
        ext = MetricExtractor([], [])
        bits = ext.peaks_to_key_bits()
        assert len(bits) == 256
        assert all(b == 0 for b in bits)

    def test_hemisphere_mapping(self):
        # Peak with theta < pi/2 should map to 0
        # Peak with theta > pi/2 should map to 1
        peaks = [
            Peak(grid_index=(0, 0, 0), r=0.5, theta=0.5, phi=0.0,
                 amplitude=1.0, energy=1.0),
            Peak(grid_index=(0, 0, 0), r=0.5, theta=2.0, phi=0.0,
                 amplitude=1.0, energy=1.0),
        ]
        ext = MetricExtractor(peaks, [])
        bits = ext.peaks_to_key_bits()
        assert bits[0] == 0  # theta=0.5 < pi/2
        assert bits[1] == 1  # theta=2.0 > pi/2


class TestFullReport:
    def test_report_keys(self):
        ext = MetricExtractor(make_peaks(5), make_history(10))
        report = ext.full_report()
        assert "peak_stats" in report
        assert "thread_stats" in report
        assert "extracted_bits_count" in report
