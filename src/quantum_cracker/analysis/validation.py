"""Statistical validation of extracted results against ground truth."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import binom

if TYPE_CHECKING:
    from quantum_cracker.core.key_interface import KeyInput


class Validator:
    """Compare extracted results against ground truth key.

    Computes bit match rates, confidence intervals, and ghost harmonic counts.
    """

    def __init__(self, original_key: KeyInput, extracted_bits: list[int]) -> None:
        self.original = original_key
        self.extracted = extracted_bits

    def bit_match_rate(self) -> float:
        """Fraction of bits that match (0.0 to 1.0)."""
        original_bits = self.original.as_bits
        n = min(len(original_bits), len(self.extracted))
        if n == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(original_bits[:n], self.extracted[:n]))
        return matches / n

    def bit_matches(self) -> list[bool]:
        """Per-bit match/mismatch list."""
        original_bits = self.original.as_bits
        n = min(len(original_bits), len(self.extracted))
        return [
            original_bits[i] == self.extracted[i]
            for i in range(n)
        ]

    def confidence_interval(self, alpha: float = 0.95) -> tuple[float, float]:
        """Binomial confidence interval on bit match rate.

        Returns (lower, upper) bounds as fractions in [0, 1].
        """
        n = min(len(self.original.as_bits), len(self.extracted))
        if n == 0:
            return (0.0, 0.0)
        k = int(self.bit_match_rate() * n)
        p_hat = k / n
        lo, hi = binom.interval(alpha, n, p_hat)
        return (float(lo) / n, float(hi) / n)

    def peak_alignment_score(self, peaks_theta: list[float] | None = None) -> float:
        """How well extracted peak angular positions align with expected.

        Computes the mean absolute deviation of peak thetas from the
        expected hemisphere boundaries (0 for bit=0, pi for bit=1).

        Returns a score in [0, 1] where 1 is perfect alignment.
        """
        if peaks_theta is None or len(peaks_theta) == 0:
            return 0.0

        original_bits = self.original.as_bits
        n = min(len(peaks_theta), len(original_bits))

        deviations = []
        for i in range(n):
            expected_center = 0.0 if original_bits[i] == 0 else np.pi
            deviation = abs(peaks_theta[i] - expected_center) / np.pi
            deviations.append(deviation)

        mean_deviation = np.mean(deviations)
        return float(1.0 - mean_deviation)

    def ghost_harmonic_count(self, total_peaks: int = 0, expected_peaks: int = 78) -> int:
        """Count false peaks (ghost harmonics).

        Ghost harmonics are peaks beyond the expected count that are
        artifacts of the 78 MHz vibration.
        """
        return max(0, total_peaks - expected_peaks)

    def summary(self, total_peaks: int = 0, peaks_theta: list[float] | None = None) -> dict:
        """Full validation summary."""
        return {
            "bit_match_rate": self.bit_match_rate(),
            "peak_alignment": self.peak_alignment_score(peaks_theta),
            "confidence_interval": self.confidence_interval(),
            "ghost_count": self.ghost_harmonic_count(total_peaks),
        }
