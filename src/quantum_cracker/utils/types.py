"""Dataclass definitions for the Quantum Cracker simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    grid_size: int = 78
    num_threads: int = 256
    resonance_freq: float = 78.0
    resonance_strength: float = 0.05
    timesteps: int = 1000
    dt: float = 0.01
    expansion_rate: float = 1.616255e-35  # Planck length per tick


@dataclass
class ThreadState:
    """State of a single expansion thread."""

    index: int
    direction: NDArray[np.float64]  # (3,) unit vector
    position: NDArray[np.float64]  # (3,) current position
    gap: float  # angular gap to nearest neighbor (radians)
    visible: bool  # gap * radius > observable threshold


@dataclass
class Peak:
    """A resonance peak extracted from the voxel grid."""

    grid_index: tuple[int, int, int]  # (ir, itheta, iphi)
    r: float
    theta: float
    phi: float
    amplitude: float
    energy: float


@dataclass
class SimulationResult:
    """Complete results from a simulation run."""

    key_hex: str
    peaks: list[Peak] = field(default_factory=list)
    rip_history: list[dict] = field(default_factory=list)  # type: ignore[type-arg]
    final_visible_count: int = 0
    total_steps: int = 0
    metadata: dict = field(default_factory=dict)  # type: ignore[type-arg]


@dataclass
class AnalysisResult:
    """Results from the analysis layer."""

    bit_match_rate: float = 0.0
    peak_alignment_score: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    ghost_harmonic_count: int = 0
    peak_stats: dict = field(default_factory=dict)  # type: ignore[type-arg]
    thread_stats: dict = field(default_factory=dict)  # type: ignore[type-arg]
