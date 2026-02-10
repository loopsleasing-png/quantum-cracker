"""256-thread Rip Engine: source-expansion simulator.

Models atoms as point sources emitting 256 Planck-length threads.
At Planck density, threads "rip" outward. When the angular gap between
threads exceeds the observable threshold (400nm), they become visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from quantum_cracker.utils.constants import (
    OBSERVABLE_THRESHOLD_M,
    PLANCK_LENGTH,
)
from quantum_cracker.utils.math_helpers import (
    nearest_neighbor_gaps,
    uniform_sphere_points,
)
from quantum_cracker.utils.types import SimulationConfig, ThreadState

if TYPE_CHECKING:
    from quantum_cracker.core.key_interface import KeyInput


class RipEngine:
    """Simulate 256-thread rip expansion from Planck scale to observable threshold.

    Each thread is a unit vector direction emanating from the origin.
    All threads expand at the same radius. The angular gap between
    nearest neighbors determines when each thread becomes "visible"
    (gap * radius > 400nm).
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.num_threads = self.config.num_threads

        # State -- initialized to defaults, call initialize_* to set up
        self.directions: NDArray[np.float64] = np.zeros((self.num_threads, 3))
        self.radius: float = PLANCK_LENGTH
        self.gaps: NDArray[np.float64] = np.zeros(self.num_threads)
        self.visible: NDArray[np.bool_] = np.zeros(self.num_threads, dtype=bool)

        self.tick: int = 0
        self.history: list[dict] = []

    @property
    def positions(self) -> NDArray[np.float64]:
        """Current positions of all threads (direction * radius)."""
        return self.directions * self.radius

    @property
    def num_visible(self) -> int:
        """How many threads have crossed the observable threshold."""
        return int(np.sum(self.visible))

    @property
    def all_visible(self) -> bool:
        """True when all threads are individually observable."""
        return bool(np.all(self.visible))

    @property
    def avg_gap(self) -> float:
        """Average angular gap across all threads."""
        return float(np.mean(self.gaps)) if len(self.gaps) > 0 else 0.0

    def initialize_from_key(self, key: KeyInput) -> None:
        """Set thread directions from a 256-bit key."""
        self.directions = key.to_thread_directions()
        self.radius = PLANCK_LENGTH
        self.tick = 0
        self.history = []
        self._compute_gaps()
        self._update_visibility()

    def initialize_random(self) -> None:
        """Initialize with uniform random spherical distribution."""
        self.directions = uniform_sphere_points(self.num_threads)
        self.radius = PLANCK_LENGTH
        self.tick = 0
        self.history = []
        self._compute_gaps()
        self._update_visibility()

    def step(self, dt: float = 1.0) -> None:
        """Advance one tick.

        The radius expands multiplicatively (cosmological expansion model).
        After position update, recompute gaps and visibility.
        """
        self.tick += 1
        self.radius *= 1.0 + dt * self.config.expansion_rate
        self._update_visibility()
        self._record_history()

    def run(self, num_steps: int, dt: float = 1.0) -> list[dict]:
        """Run simulation for num_steps ticks. Return history."""
        for _ in range(num_steps):
            self.step(dt)
        return self.history

    def run_until_visible(self, max_steps: int = 10_000_000, dt: float = 1.0) -> list[dict]:
        """Run until all threads are visible or max_steps reached."""
        for _ in range(max_steps):
            self.step(dt)
            if self.all_visible:
                break
        return self.history

    def _compute_gaps(self) -> None:
        """Compute angular gap between each thread and its nearest neighbor.

        Uses vectorized pairwise angular distance matrix.
        For 256 threads this is a 256x256 matrix -- fast in numpy.
        """
        self.gaps = nearest_neighbor_gaps(self.directions)

    def _update_visibility(self) -> None:
        """Mark threads as visible when gap * radius > observable threshold.

        The angular gap times the current radius gives the physical
        separation at the sphere surface.
        """
        physical_gaps = self.gaps * self.radius
        self.visible = physical_gaps > OBSERVABLE_THRESHOLD_M

    def _record_history(self) -> None:
        """Snapshot current state into history list."""
        self.history.append({
            "tick": self.tick,
            "radius": self.radius,
            "avg_gap": self.avg_gap,
            "min_gap": float(np.min(self.gaps)),
            "max_gap": float(np.max(self.gaps)),
            "num_visible": self.num_visible,
            "all_visible": self.all_visible,
            "min_physical_gap": float(np.min(self.gaps) * self.radius),
            "max_physical_gap": float(np.max(self.gaps) * self.radius),
        })

    def get_thread_state(self, index: int) -> ThreadState:
        """Return detailed state for a single thread."""
        return ThreadState(
            index=index,
            direction=self.directions[index].copy(),
            position=self.positions[index].copy(),
            gap=float(self.gaps[index]),
            visible=bool(self.visible[index]),
        )

    def get_all_thread_states(self) -> list[ThreadState]:
        """Return state for all threads."""
        return [self.get_thread_state(i) for i in range(self.num_threads)]
