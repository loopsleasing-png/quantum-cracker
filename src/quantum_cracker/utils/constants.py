"""Physical and mathematical constants for the Quantum Cracker simulation."""

import numpy as np

# -- Simulation grid --
GRID_SIZE: int = 78
NUM_THREADS: int = 256

# -- Physics --
PLANCK_LENGTH: float = 1.616255e-35  # meters
PLANCK_DENSITY: float = 5.155e96  # kg/m^3
SPEED_OF_LIGHT: float = 2.998e8  # m/s
HBAR: float = 1.054571817e-34  # reduced Planck constant, J*s

# -- Resonance --
RESONANCE_FREQ_MHZ: float = 78.0
RESONANCE_FREQ_HZ: float = RESONANCE_FREQ_MHZ * 1e6

# -- Spherical harmonic degree --
SH_DEGREE: int = 78

# -- Observation --
OBSERVABLE_THRESHOLD_NM: float = 400.0  # nanometers (violet light boundary)
OBSERVABLE_THRESHOLD_M: float = OBSERVABLE_THRESHOLD_NM * 1e-9

# -- Derived --
GRID_VOXEL_COUNT: int = GRID_SIZE ** 3  # 474,552
GOLDEN_RATIO: float = (1.0 + np.sqrt(5.0)) / 2.0
