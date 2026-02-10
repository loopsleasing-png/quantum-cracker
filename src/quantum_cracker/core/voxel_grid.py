"""78x78x78 Spherical Voxel Grid for the Quantum Cracker simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import sph_harm_y

from quantum_cracker.utils.constants import GRID_SIZE
from quantum_cracker.utils.math_helpers import spherical_to_cartesian

if TYPE_CHECKING:
    from quantum_cracker.core.key_interface import KeyInput


class SphericalVoxelGrid:
    """3D grid in spherical coordinates mapped to Cartesian.

    Indices: [r_index, theta_index, phi_index]
    where r in [0, 1], theta in [0, pi], phi in [0, 2*pi].
    """

    def __init__(self, size: int = GRID_SIZE) -> None:
        self.size = size

        # Coordinate arrays
        self.r_coords = np.linspace(0.01, 1.0, size)  # avoid r=0 singularity
        self.theta_coords = np.linspace(0.01, np.pi - 0.01, size)  # avoid poles
        self.phi_coords = np.linspace(0, 2 * np.pi, size, endpoint=False)

        # State arrays -- shape (size, size, size)
        self.amplitude = np.zeros((size, size, size), dtype=np.float64)
        self.phase = np.zeros((size, size, size), dtype=np.float64)
        self.energy = np.zeros((size, size, size), dtype=np.float64)

        # Cache
        self._cartesian_cache: NDArray[np.float64] | None = None

    def initialize_from_key(self, key: KeyInput) -> None:
        """Set initial grid state from a 256-bit key."""
        self.amplitude = key.to_grid_state(self.size)
        self.phase = np.zeros_like(self.amplitude)
        self.energy = np.abs(self.amplitude) ** 2

    def reset(self) -> None:
        """Zero out all state arrays."""
        self.amplitude[:] = 0.0
        self.phase[:] = 0.0
        self.energy[:] = 0.0
        self._cartesian_cache = None

    def get_cartesian_coords(self) -> NDArray[np.float64]:
        """Return (N^3, 3) array of Cartesian positions for all voxels.

        Results are cached after first computation.
        """
        if self._cartesian_cache is not None:
            return self._cartesian_cache

        r, theta, phi = np.meshgrid(
            self.r_coords, self.theta_coords, self.phi_coords, indexing="ij"
        )
        x, y, z = spherical_to_cartesian(r.ravel(), theta.ravel(), phi.ravel())
        self._cartesian_cache = np.stack([x, y, z], axis=1)
        return self._cartesian_cache

    def get_voxel(self, ir: int, itheta: int, iphi: int) -> dict:
        """Query state of a single voxel by grid indices."""
        return {
            "amplitude": float(self.amplitude[ir, itheta, iphi]),
            "phase": float(self.phase[ir, itheta, iphi]),
            "energy": float(self.energy[ir, itheta, iphi]),
            "r": float(self.r_coords[ir]),
            "theta": float(self.theta_coords[itheta]),
            "phi": float(self.phi_coords[iphi]),
        }

    def decompose_spherical_harmonics(
        self, l_max: int = 78
    ) -> NDArray[np.float64]:
        """Decompose amplitude field into spherical harmonic coefficients.

        For each radial shell, project the angular amplitude onto SH basis
        functions up to degree l_max using numerical integration.

        Returns:
            Coefficient array of shape (size, l_max+1, 2*l_max+1)
            indexed as [r_index, l, m + l_max].
        """
        coeffs = np.zeros(
            (self.size, l_max + 1, 2 * l_max + 1), dtype=np.float64
        )

        theta = self.theta_coords
        phi = self.phi_coords
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        # Integration weight: sin(theta) * dtheta * dphi
        dtheta = theta[1] - theta[0] if len(theta) > 1 else 1.0
        dphi = phi[1] - phi[0] if len(phi) > 1 else 1.0
        weight = np.sin(theta_grid) * dtheta * dphi

        for degree in range(min(l_max + 1, self.size)):
            for m in range(-degree, degree + 1):
                ylm = sph_harm_y(degree, m, theta_grid, phi_grid).real
                for ir in range(self.size):
                    shell = self.amplitude[ir, :, :]
                    coeffs[ir, degree, m + l_max] = np.sum(shell * ylm * weight)

        return coeffs

    def reconstruct_from_sh(
        self, coeffs: NDArray[np.float64], l_max: int = 78
    ) -> None:
        """Reconstruct the amplitude field from SH coefficients.

        Sets self.amplitude from the provided coefficient array.
        """
        theta = self.theta_coords
        phi = self.phi_coords
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        for ir in range(self.size):
            shell = np.zeros((self.size, self.size), dtype=np.float64)
            for degree in range(min(l_max + 1, self.size)):
                for m in range(-degree, degree + 1):
                    c = coeffs[ir, degree, m + l_max]
                    if abs(c) > 1e-15:
                        ylm = sph_harm_y(degree, m, theta_grid, phi_grid).real
                        shell += c * ylm
            self.amplitude[ir, :, :] = shell

        self.energy = np.abs(self.amplitude) ** 2

    def snapshot(self) -> dict:
        """Return serializable snapshot of current state."""
        return {
            "size": self.size,
            "amplitude_sum": float(np.sum(self.amplitude)),
            "energy_sum": float(np.sum(self.energy)),
            "energy_max": float(np.max(self.energy)),
            "nonzero_voxels": int(np.count_nonzero(self.amplitude)),
        }
