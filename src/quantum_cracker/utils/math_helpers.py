"""Coordinate transforms and math utilities for the Quantum Cracker simulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantum_cracker.utils.constants import GOLDEN_RATIO


def spherical_to_cartesian(
    r: float | NDArray[np.float64],
    theta: float | NDArray[np.float64],
    phi: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert spherical (r, theta, phi) to Cartesian (x, y, z).

    theta: polar angle [0, pi] (from +z axis)
    phi: azimuthal angle [0, 2*pi] (from +x axis in xy-plane)
    """
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian_to_spherical(
    x: float | NDArray[np.float64],
    y: float | NDArray[np.float64],
    z: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert Cartesian (x, y, z) to spherical (r, theta, phi).

    Returns:
        r: radius >= 0
        theta: polar angle [0, pi]
        phi: azimuthal angle [0, 2*pi]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.where(r > 0, np.arccos(np.clip(z / r, -1.0, 1.0)), 0.0)
    phi = np.arctan2(y, x) % (2 * np.pi)
    return r, theta, phi


def uniform_sphere_points(n: int) -> NDArray[np.float64]:
    """Generate n approximately uniformly distributed points on the unit sphere.

    Uses the Fibonacci/golden spiral method for even spacing.

    Returns:
        ndarray of shape (n, 3) -- unit vectors in Cartesian coordinates.
    """
    indices = np.arange(n, dtype=np.float64)

    # z goes from ~1 to ~-1 uniformly
    z = 1.0 - (2.0 * indices + 1.0) / n
    r_xy = np.sqrt(1.0 - z**2)

    # Golden angle spacing for phi
    phi = 2.0 * np.pi * indices / GOLDEN_RATIO

    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)

    return np.stack([x, y, z], axis=1)


def angular_gap(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
    """Compute the angle (radians) between two vectors.

    Works with any-dimensional vectors. Returns value in [0, pi].
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))


def normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a vector to unit length. Handles zero vectors gracefully."""
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    return v / norm


def pairwise_angular_distances(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute pairwise angular distances between unit vectors.

    Args:
        vectors: (n, 3) array of unit vectors.

    Returns:
        (n, n) symmetric matrix of angles in radians.
    """
    dots = vectors @ vectors.T
    np.clip(dots, -1.0, 1.0, out=dots)
    return np.arccos(dots)


def nearest_neighbor_gaps(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """For each vector, find the angular gap to its nearest neighbor.

    Args:
        vectors: (n, 3) array of unit vectors.

    Returns:
        (n,) array of minimum angular gaps in radians.
    """
    dists = pairwise_angular_distances(vectors)
    # Set diagonal to infinity so a vector doesn't match itself
    np.fill_diagonal(dists, np.inf)
    return np.min(dists, axis=1)
