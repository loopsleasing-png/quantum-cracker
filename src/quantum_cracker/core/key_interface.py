"""Key input interface for accepting, validating, and converting 256-bit keys."""

from __future__ import annotations

import secrets

import numpy as np
from numpy.typing import NDArray
from scipy.special import sph_harm_y

from quantum_cracker.utils.constants import GRID_SIZE, NUM_THREADS
from quantum_cracker.utils.math_helpers import uniform_sphere_points


class KeyInput:
    """Accept, validate, and convert 256-bit keys.

    Supports hex strings (64 chars), binary strings (256 chars),
    integers, and raw bytes (32 bytes).
    """

    def __init__(self, key: str | int | bytes) -> None:
        if isinstance(key, int):
            if key < 0 or key.bit_length() > 256:
                raise ValueError("Integer key must be in range [0, 2^256 - 1]")
            self._value = key

        elif isinstance(key, bytes):
            if len(key) != 32:
                raise ValueError(f"Bytes key must be exactly 32 bytes, got {len(key)}")
            self._value = int.from_bytes(key, byteorder="big")

        elif isinstance(key, str):
            cleaned = key.strip()
            if len(cleaned) == 64:
                # Hex string
                try:
                    self._value = int(cleaned, 16)
                except ValueError:
                    raise ValueError("Invalid hex string")
            elif len(cleaned) == 256 and all(c in "01" for c in cleaned):
                # Binary string
                self._value = int(cleaned, 2)
            elif cleaned.startswith("0x"):
                try:
                    self._value = int(cleaned, 16)
                    if self._value.bit_length() > 256:
                        raise ValueError("Hex value exceeds 256 bits")
                except ValueError:
                    raise ValueError("Invalid hex string")
            else:
                raise ValueError(
                    "String key must be 64-char hex or 256-char binary, "
                    f"got length {len(cleaned)}"
                )
        else:
            raise TypeError(f"Key must be str, int, or bytes, got {type(key).__name__}")

    @property
    def as_int(self) -> int:
        return self._value

    @property
    def as_hex(self) -> str:
        return f"{self._value:064x}"

    @property
    def as_bits(self) -> list[int]:
        """Return list of 256 individual bits (MSB first)."""
        return [int(b) for b in f"{self._value:0256b}"]

    @property
    def as_bytes(self) -> bytes:
        return self._value.to_bytes(32, byteorder="big")

    def to_thread_directions(self) -> NDArray[np.float64]:
        """Convert 256 bits to 256 unit vectors on the sphere.

        Uses Fibonacci spiral for base positions. Each bit flips the z-component
        of the corresponding vector: bit=0 keeps original, bit=1 flips z.
        This creates a key-dependent distribution on the sphere.

        Returns:
            ndarray of shape (256, 3) -- unit vectors in Cartesian.
        """
        base_points = uniform_sphere_points(NUM_THREADS)
        bits = np.array(self.as_bits, dtype=np.float64)

        # Flip z for bits that are 1
        z_flip = np.where(bits == 1, -1.0, 1.0)
        base_points[:, 2] *= z_flip

        # Re-normalize (flipping z doesn't change norm for unit vectors,
        # but be safe)
        norms = np.linalg.norm(base_points, axis=1, keepdims=True)
        base_points /= norms

        return base_points

    def to_grid_state(self, grid_size: int = GRID_SIZE) -> NDArray[np.float64]:
        """Map 256 bits into a (grid_size, grid_size, grid_size) amplitude array.

        Distributes key bits as coefficients of spherical harmonics evaluated
        on the grid. Uses SH orders (l, m) mapped from bit indices, with the
        bit value (+1 or -1) as the coefficient.

        Returns:
            ndarray of shape (grid_size, grid_size, grid_size).
        """
        bits = np.array(self.as_bits, dtype=np.float64)
        # Map bits 0/1 -> -1/+1 as SH coefficients
        coeffs = 2.0 * bits - 1.0

        theta = np.linspace(0, np.pi, grid_size)
        phi = np.linspace(0, 2 * np.pi, grid_size)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        # Sum the first 256 spherical harmonics weighted by key bits
        # Map bit index i to (l, m): l goes 0,1,1,1,2,2,2,2,2,...
        # For l, there are 2l+1 values of m, so cumulative count = (l+1)^2
        angular_field = np.zeros((grid_size, grid_size), dtype=np.float64)
        bit_idx = 0
        degree = 0
        while bit_idx < NUM_THREADS:
            for m in range(-degree, degree + 1):
                if bit_idx >= NUM_THREADS:
                    break
                ylm = sph_harm_y(degree, m, theta_grid, phi_grid).real
                angular_field += coeffs[bit_idx] * ylm
                bit_idx += 1
            degree += 1

        # Normalize to [-1, 1] range
        max_val = np.abs(angular_field).max()
        if max_val > 0:
            angular_field /= max_val

        # Expand to 3D: modulate by radial Gaussian
        r = np.linspace(0, 1, grid_size)
        radial = np.exp(-((r - 0.5) ** 2) / 0.1)

        # Outer product: (grid_size,) x (grid_size, grid_size) -> (grid_size, grid_size, grid_size)
        grid = radial[:, np.newaxis, np.newaxis] * angular_field[np.newaxis, :, :]

        return grid

    @staticmethod
    def random() -> KeyInput:
        """Generate a random 256-bit key."""
        return KeyInput(secrets.token_bytes(32))

    @staticmethod
    def from_cli() -> KeyInput:
        """Interactive CLI prompt for key input."""
        print("Enter a 256-bit key:")
        print("  Formats: 64-char hex, 256-char binary, or 'random'")
        raw = input("> ").strip()
        if raw.lower() == "random":
            key = KeyInput.random()
            print(f"Generated: {key.as_hex}")
            return key
        return KeyInput(raw)

    def __repr__(self) -> str:
        return f"KeyInput(0x{self.as_hex[:16]}...)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KeyInput):
            return NotImplemented
        return self._value == other._value
