"""Harmonic Compiler: 78 MHz resonance filter and key extraction.

Applies the 78 MHz resonant vibration to the spherical voxel grid
using the equation: vibration = sin(78 * phi + t) * cos(78 * theta).
Filters through spherical harmonics at degree l=78 and extracts
the 78 peak nodes that represent the resolved key fragments.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter
from scipy.special import sph_harm_y

from quantum_cracker.utils.constants import GRID_SIZE, RESONANCE_FREQ_MHZ, SH_DEGREE
from quantum_cracker.utils.types import Peak, SimulationConfig

from quantum_cracker.core.voxel_grid import SphericalVoxelGrid


class HarmonicCompiler:
    """Apply harmonic resonance to the spherical voxel grid and extract peaks.

    The compiler iterates a resonance vibration on the grid, optionally
    filters through spherical harmonics at l=78, and extracts the top
    resonance peaks.
    """

    def __init__(
        self,
        grid: SphericalVoxelGrid,
        config: SimulationConfig | None = None,
    ) -> None:
        self.grid = grid
        self.config = config or SimulationConfig()
        self.time: float = 0.0
        self.peaks: list[Peak] = []
        self._vibration_cache: NDArray[np.float64] | None = None

    def _build_vibration_field(self, t: float) -> NDArray[np.float64]:
        """Compute the resonance vibration field at time t.

        vibration[r, theta, phi] = sin(78 * phi + t) * cos(78 * theta)

        This is the Python equivalent of the GLSL vertex shader.
        """
        freq = RESONANCE_FREQ_MHZ
        _, theta_grid, phi_grid = np.meshgrid(
            self.grid.r_coords,
            self.grid.theta_coords,
            self.grid.phi_coords,
            indexing="ij",
        )
        return np.sin(freq * phi_grid + t) * np.cos(freq * theta_grid)

    def apply_resonance(self, t: float | None = None) -> None:
        """Apply 78 MHz resonant vibration to the grid.

        Modulates the amplitude field multiplicatively using the resonance
        strength parameter.
        """
        if t is not None:
            self.time = t

        vibration = self._build_vibration_field(self.time)
        strength = self.config.resonance_strength
        self.grid.amplitude *= 1.0 + vibration * strength
        self.grid.energy = np.abs(self.grid.amplitude) ** 2

    def apply_spherical_harmonic_filter(self, l_target: int = SH_DEGREE) -> None:
        """Filter grid through spherical harmonics, keeping only degree l_target.

        For each radial shell:
        1. Project amplitude onto SH basis at degree l_target
        2. Reconstruct from only those coefficients

        This isolates the l=78 resonant mode -- the "compiler" step.
        """
        theta = self.grid.theta_coords
        phi = self.grid.phi_coords
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        dtheta = theta[1] - theta[0] if len(theta) > 1 else 1.0
        dphi = phi[1] - phi[0] if len(phi) > 1 else 1.0
        weight = np.sin(theta_grid) * dtheta * dphi

        # Precompute basis functions for degree l_target
        basis_functions = []
        for m in range(-l_target, l_target + 1):
            if abs(m) <= l_target:
                ylm = sph_harm_y(l_target, m, theta_grid, phi_grid).real
                basis_functions.append((m, ylm))

        # For each radial shell, project and reconstruct
        for ir in range(self.grid.size):
            shell = self.grid.amplitude[ir, :, :]

            # Project onto basis
            reconstructed = np.zeros_like(shell)
            for _m, ylm in basis_functions:
                coeff = np.sum(shell * ylm * weight)
                reconstructed += coeff * ylm

            self.grid.amplitude[ir, :, :] = reconstructed

        self.grid.energy = np.abs(self.grid.amplitude) ** 2

    def compute_hamiltonian_eigenvalues(
        self, shell_index: int = -1, l_max: int = 20
    ) -> NDArray[np.float64]:
        """Compute eigenvalues of a reduced Hamiltonian on a single shell.

        H = T + V_lattice + V_compiler
        Uses a finite-difference Laplacian on the angular grid for kinetic
        energy, the amplitude as lattice potential, and the resonance field
        as the compiler term.

        Returns eigenvalues sorted ascending (ground state first).
        """
        n = self.grid.size
        shell = self.grid.amplitude[shell_index, :, :]

        # Flatten the angular shell to build the Hamiltonian matrix
        n_angular = n * n
        H = np.zeros((n_angular, n_angular), dtype=np.float64)

        # Kinetic energy: finite-difference Laplacian on (theta, phi) grid
        dtheta = self.grid.theta_coords[1] - self.grid.theta_coords[0]
        dphi = self.grid.phi_coords[1] - self.grid.phi_coords[0]

        for i in range(n):
            for j in range(n):
                idx = i * n + j
                H[idx, idx] = -2.0 / dtheta**2 - 2.0 / dphi**2

                # theta neighbors
                if i > 0:
                    H[idx, (i - 1) * n + j] = 1.0 / dtheta**2
                if i < n - 1:
                    H[idx, (i + 1) * n + j] = 1.0 / dtheta**2

                # phi neighbors (periodic)
                H[idx, i * n + (j - 1) % n] += 1.0 / dphi**2
                H[idx, i * n + (j + 1) % n] += 1.0 / dphi**2

        # Scale kinetic energy
        H *= -0.5  # -hbar^2 / 2m in natural units

        # Lattice potential: diagonal from amplitude field
        V_lattice = shell.ravel()
        np.fill_diagonal(H, H.diagonal() + V_lattice)

        # Compiler term: resonance vibration on this shell
        vibration = self._build_vibration_field(self.time)[shell_index, :, :].ravel()
        np.fill_diagonal(H, H.diagonal() + vibration * self.config.resonance_strength)

        # Compute eigenvalues (only need a few lowest)
        eigenvalues = np.linalg.eigvalsh(H)
        return eigenvalues

    def extract_peaks(self, num_peaks: int = 78) -> list[Peak]:
        """Find the top N resonance peaks in the energy field.

        A peak is a local maximum in the 3D energy array, detected using
        scipy.ndimage.maximum_filter.
        """
        energy = self.grid.energy

        # Find local maxima using 3x3x3 neighborhood
        local_max = maximum_filter(energy, size=3)
        threshold = np.percentile(energy[energy > 0], 50) if np.any(energy > 0) else 0.0
        peaks_mask = (energy == local_max) & (energy > threshold)

        peak_indices = np.argwhere(peaks_mask)
        peak_energies = energy[peaks_mask]

        if len(peak_energies) == 0:
            self.peaks = []
            return self.peaks

        # Sort by energy descending, take top num_peaks
        top_order = np.argsort(peak_energies)[::-1][:num_peaks]

        self.peaks = []
        for idx in top_order:
            ir, itheta, iphi = peak_indices[idx]
            self.peaks.append(
                Peak(
                    grid_index=(int(ir), int(itheta), int(iphi)),
                    r=float(self.grid.r_coords[ir]),
                    theta=float(self.grid.theta_coords[itheta]),
                    phi=float(self.grid.phi_coords[iphi]),
                    amplitude=float(self.grid.amplitude[ir, itheta, iphi]),
                    energy=float(energy[ir, itheta, iphi]),
                )
            )

        return self.peaks

    def compile(
        self,
        num_steps: int = 100,
        dt: float = 0.01,
        apply_sh_filter: bool = False,
        sh_filter_interval: int = 10,
    ) -> list[Peak]:
        """Full compilation pipeline.

        1. Iterate resonance at each timestep
        2. Optionally apply SH filter at intervals
        3. Extract peaks at the end

        Returns the extracted peaks.
        """
        for step in range(num_steps):
            self.time += dt
            self.apply_resonance(self.time)

            if apply_sh_filter and (step + 1) % sh_filter_interval == 0:
                self.apply_spherical_harmonic_filter()

        return self.extract_peaks()
