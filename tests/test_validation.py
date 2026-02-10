"""Tests for statistical validation."""

import pytest

from quantum_cracker.analysis.validation import Validator
from quantum_cracker.core.key_interface import KeyInput


class TestBitMatchRate:
    def test_perfect_match(self):
        key = KeyInput(42)
        bits = key.as_bits
        val = Validator(key, bits)
        assert val.bit_match_rate() == 1.0

    def test_all_wrong(self):
        key = KeyInput(0)
        # All bits are 0, flip all to 1
        wrong_bits = [1] * 256
        val = Validator(key, wrong_bits)
        # Key=0 means all 256 bits are 0; wrong_bits are all 1
        assert val.bit_match_rate() == 0.0

    def test_half_match(self):
        key = KeyInput(0)
        # First 128 correct (0), last 128 wrong (1)
        half_bits = [0] * 128 + [1] * 128
        val = Validator(key, half_bits)
        assert val.bit_match_rate() == pytest.approx(0.5)

    def test_empty_extracted(self):
        key = KeyInput(42)
        val = Validator(key, [])
        assert val.bit_match_rate() == 0.0


class TestBitMatches:
    def test_returns_list_of_bools(self):
        key = KeyInput(42)
        bits = key.as_bits
        val = Validator(key, bits)
        matches = val.bit_matches()
        assert len(matches) == 256
        assert all(m is True for m in matches)

    def test_mismatch_detection(self):
        key = KeyInput(0)
        wrong = [1] + [0] * 255
        val = Validator(key, wrong)
        matches = val.bit_matches()
        assert matches[0] is False
        assert all(m is True for m in matches[1:])


class TestConfidenceInterval:
    def test_perfect_match_ci(self):
        key = KeyInput(42)
        val = Validator(key, key.as_bits)
        lo, hi = val.confidence_interval()
        assert lo >= 0.9
        assert hi <= 1.0

    def test_ci_bounds(self):
        key = KeyInput(0)
        val = Validator(key, [0] * 128 + [1] * 128)
        lo, hi = val.confidence_interval()
        assert 0.0 <= lo <= hi <= 1.0

    def test_empty_ci(self):
        key = KeyInput(42)
        val = Validator(key, [])
        assert val.confidence_interval() == (0.0, 0.0)


class TestPeakAlignment:
    def test_perfect_alignment(self):
        key = KeyInput(0)
        # All bits are 0, so expected theta near 0
        thetas = [0.1] * 256
        val = Validator(key, [0] * 256)
        score = val.peak_alignment_score(thetas)
        assert score > 0.9

    def test_no_thetas(self):
        key = KeyInput(42)
        val = Validator(key, [0] * 256)
        assert val.peak_alignment_score() == 0.0

    def test_score_range(self):
        key = KeyInput.random()
        thetas = [1.5] * 256
        val = Validator(key, [0] * 256)
        score = val.peak_alignment_score(thetas)
        assert 0.0 <= score <= 1.0


class TestGhostHarmonics:
    def test_no_ghosts(self):
        key = KeyInput(42)
        val = Validator(key, [0] * 256)
        assert val.ghost_harmonic_count(total_peaks=78) == 0

    def test_with_ghosts(self):
        key = KeyInput(42)
        val = Validator(key, [0] * 256)
        assert val.ghost_harmonic_count(total_peaks=100) == 22

    def test_fewer_than_expected(self):
        key = KeyInput(42)
        val = Validator(key, [0] * 256)
        assert val.ghost_harmonic_count(total_peaks=50) == 0


class TestSummary:
    def test_summary_keys(self):
        key = KeyInput(42)
        val = Validator(key, key.as_bits)
        summary = val.summary()
        assert "bit_match_rate" in summary
        assert "peak_alignment" in summary
        assert "confidence_interval" in summary
        assert "ghost_count" in summary
