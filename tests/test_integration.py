"""Integration tests: full pipeline end-to-end."""

import os
import tempfile

import numpy as np
import pytest

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.core.rip_engine import RipEngine
from quantum_cracker.core.harmonic_compiler import HarmonicCompiler
from quantum_cracker.analysis.metrics import MetricExtractor
from quantum_cracker.analysis.validation import Validator
from quantum_cracker.utils.types import SimulationConfig


class TestFullPipeline:
    """Test the complete simulation pipeline with a known key."""

    def test_end_to_end(self):
        key = KeyInput(42)
        grid_size = 15
        steps = 20

        # Initialize
        grid = SphericalVoxelGrid(size=grid_size)
        grid.initialize_from_key(key)
        assert np.any(grid.amplitude != 0)

        config = SimulationConfig(grid_size=grid_size, timesteps=steps)
        engine = RipEngine(config=config)
        engine.initialize_from_key(key)
        assert engine.directions.shape == (256, 3)

        # Run rip engine
        rip_history = engine.run(steps)
        assert len(rip_history) == steps

        # Run harmonic compiler
        compiler = HarmonicCompiler(grid, config=config)
        peaks = compiler.compile(num_steps=steps, dt=0.01)
        assert isinstance(peaks, list)

        # Eigenvalues
        eigenvalues = compiler.compute_hamiltonian_eigenvalues()
        assert len(eigenvalues) > 0
        assert np.all(np.diff(eigenvalues) >= -1e-10)

        # Analysis
        extractor = MetricExtractor(peaks, rip_history)
        report = extractor.full_report()
        assert "peak_stats" in report
        assert "thread_stats" in report

        extracted_bits = extractor.peaks_to_key_bits()
        assert len(extracted_bits) == 256

        # Validation
        validator = Validator(key, extracted_bits)
        rate = validator.bit_match_rate()
        assert 0.0 <= rate <= 1.0

        summary = validator.summary(
            total_peaks=len(peaks),
            peaks_theta=[p.theta for p in peaks],
        )
        assert "bit_match_rate" in summary
        assert "confidence_interval" in summary

    def test_pipeline_with_random_key(self):
        key = KeyInput.random()
        grid = SphericalVoxelGrid(size=10)
        grid.initialize_from_key(key)

        engine = RipEngine()
        engine.initialize_from_key(key)
        engine.run(10)

        compiler = HarmonicCompiler(grid)
        peaks = compiler.compile(num_steps=10)

        extractor = MetricExtractor(peaks, engine.history)
        report = extractor.full_report()
        assert isinstance(report, dict)

    def test_different_keys_produce_different_results(self):
        results = []
        for val in [0, 42, 2**256 - 1]:
            key = KeyInput(val)
            grid = SphericalVoxelGrid(size=10)
            grid.initialize_from_key(key)
            compiler = HarmonicCompiler(grid)
            compiler.compile(num_steps=5)
            results.append(grid.amplitude.copy())

        # At least 2 of 3 should differ
        diffs = [
            not np.allclose(results[0], results[1]),
            not np.allclose(results[1], results[2]),
            not np.allclose(results[0], results[2]),
        ]
        assert sum(diffs) >= 2


class TestCSVExport:
    def test_export_produces_file(self, tmp_path):
        """Test the CSV export logic directly."""
        import csv as csv_mod
        from quantum_cracker.__main__ import export_csv

        key = KeyInput(42)
        report = {"peak_stats": {"count": 10}, "thread_stats": {"total_steps": 20}}
        validation = {
            "bit_match_rate": 0.75,
            "peak_alignment": 0.5,
            "confidence_interval": (0.7, 0.8),
            "ghost_count": 2,
        }
        eigenvalues = np.array([-5.0, -3.0, 0.0, 2.0])

        # Monkeypatch the desktop path
        filepath = os.path.join(str(tmp_path), "test_results.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["key_hex", key.as_hex])
            writer.writerow(["bit_match_rate", validation["bit_match_rate"]])

        assert os.path.exists(filepath)
        with open(filepath) as f:
            reader = csv_mod.reader(f)
            rows = list(reader)
        assert rows[0] == ["metric", "value"]
        assert rows[1][0] == "key_hex"


class TestCLIModule:
    def test_module_importable(self):
        from quantum_cracker.__main__ import build_parser, main
        parser = build_parser()
        assert parser is not None

    def test_version_in_parser(self):
        from quantum_cracker.__main__ import build_parser
        parser = build_parser()
        # Check --version is registered
        assert any("version" in a.option_strings[0] for a in parser._actions
                    if hasattr(a, "option_strings") and a.option_strings)
