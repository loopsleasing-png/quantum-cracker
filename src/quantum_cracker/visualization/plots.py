"""Matplotlib-based 2D plots for simulation analysis."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend by default

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from quantum_cracker.core.key_interface import KeyInput
    from quantum_cracker.core.voxel_grid import SphericalVoxelGrid


class PlotSuite:
    """Matplotlib-based 2D plots for Quantum Cracker analysis."""

    def __init__(self, save_dir: str = "~/Desktop") -> None:
        self.save_dir = os.path.expanduser(save_dir)

    def _save_or_show(
        self, fig: plt.Figure, name: str, show: bool, save: bool
    ) -> plt.Figure:
        if save:
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"qc_{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def spherical_harmonic_heatmap(
        self,
        grid: SphericalVoxelGrid,
        r_index: int = -1,
        show: bool = False,
        save: bool = True,
    ) -> plt.Figure:
        """Plot amplitude on a spherical shell as a theta-phi heatmap.

        Uses Mollweide projection for the sphere unwrapping.
        """
        shell = grid.amplitude[r_index, :, :]
        theta = grid.theta_coords
        phi = grid.phi_coords

        # Mollweide projection needs longitude in [-pi, pi] and latitude in [-pi/2, pi/2]
        lon = phi - np.pi  # shift [0, 2pi] -> [-pi, pi]
        lat = np.pi / 2 - theta  # shift [0, pi] -> [pi/2, -pi/2]

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection="mollweide")
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        im = ax.pcolormesh(lon_grid, lat_grid, shell, cmap="coolwarm", shading="auto")
        fig.colorbar(im, ax=ax, label="Amplitude")
        ax.set_title(f"Spherical Harmonic Heatmap (r_index={r_index})")
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, "sh_heatmap", show, save)

    def thread_gap_vs_time(
        self,
        rip_history: list[dict],
        show: bool = False,
        save: bool = True,
    ) -> plt.Figure:
        """Plot angular gap evolution over simulation time.

        Shows avg, min, max gaps with observable threshold line.
        """
        if not rip_history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return self._save_or_show(fig, "gap_vs_time", show, save)

        ticks = [h["tick"] for h in rip_history]
        avg_gaps = [h["avg_gap"] for h in rip_history]
        min_gaps = [h["min_gap"] for h in rip_history]
        max_gaps = [h["max_gap"] for h in rip_history]
        num_visible = [h["num_visible"] for h in rip_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(ticks, avg_gaps, label="Avg gap", color="blue")
        ax1.fill_between(ticks, min_gaps, max_gaps, alpha=0.2, color="blue")
        ax1.set_ylabel("Angular Gap (radians)")
        ax1.set_title("Thread Angular Gap Evolution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(ticks, num_visible, color="green")
        ax2.axhline(y=256, color="red", linestyle="--", alpha=0.5, label="All visible (256)")
        ax2.set_xlabel("Tick")
        ax2.set_ylabel("Visible Threads")
        ax2.set_title("Visibility Progress")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return self._save_or_show(fig, "gap_vs_time", show, save)

    def energy_landscape(
        self,
        eigenvalues: NDArray[np.float64],
        show: bool = False,
        save: bool = True,
    ) -> plt.Figure:
        """Plot Hamiltonian eigenvalue spectrum."""
        fig, ax = plt.subplots(figsize=(12, 6))

        n = len(eigenvalues)
        indices = np.arange(n)

        ax.stem(indices[:50], eigenvalues[:50], linefmt="b-", markerfmt="bo", basefmt="r-")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Energy")
        ax.set_title("Hamiltonian Eigenvalue Spectrum (first 50)")
        ax.grid(True, alpha=0.3)

        # Highlight ground state
        if n > 0:
            ax.axhline(y=eigenvalues[0], color="green", linestyle="--", alpha=0.5,
                        label=f"Ground state: {eigenvalues[0]:.4f}")
        if n > 1:
            gap = eigenvalues[1] - eigenvalues[0]
            ax.annotate(
                f"Gap: {gap:.4f}",
                xy=(1, eigenvalues[1]),
                xytext=(5, eigenvalues[1] + (eigenvalues[-1] - eigenvalues[0]) * 0.1),
                arrowprops=dict(arrowstyle="->"),
            )
        ax.legend()

        return self._save_or_show(fig, "energy_landscape", show, save)

    def key_comparison(
        self,
        original: KeyInput,
        extracted: list[int],
        show: bool = False,
        save: bool = True,
    ) -> plt.Figure:
        """Visual comparison of original vs extracted key bits.

        256-wide bar showing match (green) / mismatch (red) per bit.
        """
        original_bits = original.as_bits
        n = min(len(original_bits), len(extracted))

        matches = [original_bits[i] == extracted[i] for i in range(n)]
        colors = ["#2ecc71" if m else "#e74c3c" for m in matches]

        fig, ax = plt.subplots(figsize=(16, 3))
        ax.bar(range(n), [1] * n, color=colors, width=1.0, edgecolor="none")
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Bit Index")
        ax.set_yticks([])
        match_rate = sum(matches) / n if n > 0 else 0
        ax.set_title(f"Key Comparison -- Match Rate: {match_rate:.1%} ({sum(matches)}/{n} bits)")

        # Legend
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor="#2ecc71", label="Match"),
                Patch(facecolor="#e74c3c", label="Mismatch"),
            ],
            loc="upper right",
        )

        fig.tight_layout()
        return self._save_or_show(fig, "key_comparison", show, save)

    def peak_distribution_3d(
        self,
        peaks: list,
        show: bool = False,
        save: bool = True,
    ) -> plt.Figure:
        """3D scatter of peak positions, sized by energy."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        if not peaks:
            ax.text(0, 0, 0, "No peaks")
            return self._save_or_show(fig, "peak_3d", show, save)

        # Convert spherical to Cartesian for display
        rs = np.array([p.r for p in peaks])
        thetas = np.array([p.theta for p in peaks])
        phis = np.array([p.phi for p in peaks])
        energies = np.array([p.energy for p in peaks])

        x = rs * np.sin(thetas) * np.cos(phis)
        y = rs * np.sin(thetas) * np.sin(phis)
        z = rs * np.cos(thetas)

        # Normalize energies for sizing
        if energies.max() > 0:
            sizes = 20 + 200 * energies / energies.max()
        else:
            sizes = np.full_like(energies, 50.0)

        sc = ax.scatter(x, y, z, c=energies, cmap="hot", s=sizes, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="Energy", shrink=0.6)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Peak Distribution ({len(peaks)} peaks)")

        return self._save_or_show(fig, "peak_3d", show, save)
