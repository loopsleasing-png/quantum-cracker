"""Observable metric extraction from simulation results."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantum_cracker.utils.constants import NUM_THREADS
from quantum_cracker.utils.types import Peak


class MetricExtractor:
    """Extract observable metrics from simulation results.

    Takes resonance peaks from the harmonic compiler and thread
    history from the rip engine to produce analysis reports.
    """

    def __init__(
        self,
        peaks: list[Peak],
        rip_history: list[dict],
    ) -> None:
        self.peaks = peaks
        self.rip_history = rip_history

    def resonance_peak_stats(self) -> dict:
        """Statistics on the resonance peaks.

        Returns dict with mean/std/min/max of amplitudes and energies,
        plus angular distribution metrics.
        """
        if not self.peaks:
            return {
                "count": 0,
                "amplitude_mean": 0.0,
                "amplitude_std": 0.0,
                "energy_mean": 0.0,
                "energy_std": 0.0,
                "energy_max": 0.0,
                "energy_min": 0.0,
            }

        amplitudes = np.array([p.amplitude for p in self.peaks])
        energies = np.array([p.energy for p in self.peaks])
        thetas = np.array([p.theta for p in self.peaks])
        phis = np.array([p.phi for p in self.peaks])

        return {
            "count": len(self.peaks),
            "amplitude_mean": float(np.mean(amplitudes)),
            "amplitude_std": float(np.std(amplitudes)),
            "energy_mean": float(np.mean(energies)),
            "energy_std": float(np.std(energies)),
            "energy_max": float(np.max(energies)),
            "energy_min": float(np.min(energies)),
            "theta_spread": float(np.std(thetas)),
            "phi_spread": float(np.std(phis)),
        }

    def thread_separation_stats(self) -> dict:
        """Statistics on thread separations over time from rip history.

        Returns dict with gap evolution data and visibility timeline.
        """
        if not self.rip_history:
            return {
                "total_steps": 0,
                "first_visible_tick": None,
                "all_visible_tick": None,
                "final_avg_gap": 0.0,
                "final_num_visible": 0,
            }

        first_visible_tick = None
        all_visible_tick = None
        for h in self.rip_history:
            if h["num_visible"] > 0 and first_visible_tick is None:
                first_visible_tick = h["tick"]
            if h["all_visible"] and all_visible_tick is None:
                all_visible_tick = h["tick"]

        final = self.rip_history[-1]
        return {
            "total_steps": len(self.rip_history),
            "first_visible_tick": first_visible_tick,
            "all_visible_tick": all_visible_tick,
            "final_avg_gap": final["avg_gap"],
            "final_num_visible": final["num_visible"],
            "final_radius": final["radius"],
            "final_min_physical_gap": final["min_physical_gap"],
            "final_max_physical_gap": final["max_physical_gap"],
        }

    def energy_landscape_stats(self, eigenvalues: NDArray[np.float64] | None = None) -> dict:
        """Hamiltonian eigenvalue statistics.

        If eigenvalues provided, compute ground state energy and energy gap.
        """
        if eigenvalues is None or len(eigenvalues) == 0:
            return {
                "ground_state_energy": None,
                "first_excited_energy": None,
                "energy_gap": None,
                "eigenvalue_count": 0,
            }

        return {
            "ground_state_energy": float(eigenvalues[0]),
            "first_excited_energy": float(eigenvalues[1]) if len(eigenvalues) > 1 else None,
            "energy_gap": float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else None,
            "eigenvalue_count": len(eigenvalues),
            "eigenvalue_range": float(eigenvalues[-1] - eigenvalues[0]),
        }

    def peaks_to_key_bits(self) -> list[int]:
        """Reconstruct key bits from peak positions.

        Map each peak's angular position (theta, phi) back to a bit index
        using the inverse of the key-to-grid mapping. Peaks in the
        northern hemisphere (theta < pi/2) map to bit=0, southern to bit=1.

        For positions without a peak, default to 0.

        Returns list of 256 reconstructed bits.
        """
        bits = [0] * NUM_THREADS

        if not self.peaks:
            return bits

        # Map peak positions to bit indices based on angular position
        # Distribute peaks across the 256-bit indices by their angular position
        for i, peak in enumerate(self.peaks):
            if i >= NUM_THREADS:
                break
            # Northern hemisphere -> 0, Southern -> 1
            bits[i] = 0 if peak.theta < np.pi / 2 else 1

        return bits

    def full_report(self) -> dict:
        """Aggregate all metrics into a single report."""
        return {
            "peak_stats": self.resonance_peak_stats(),
            "thread_stats": self.thread_separation_stats(),
            "extracted_bits_count": sum(self.peaks_to_key_bits()),
        }
