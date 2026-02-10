"""Tests for the utility layer: constants, types, math_helpers."""

import numpy as np
import pytest

from quantum_cracker.utils.constants import (
    GOLDEN_RATIO,
    GRID_SIZE,
    GRID_VOXEL_COUNT,
    NUM_THREADS,
    OBSERVABLE_THRESHOLD_M,
    PLANCK_LENGTH,
)
from quantum_cracker.utils.math_helpers import (
    angular_gap,
    cartesian_to_spherical,
    nearest_neighbor_gaps,
    normalize,
    pairwise_angular_distances,
    spherical_to_cartesian,
    uniform_sphere_points,
)
from quantum_cracker.utils.types import SimulationConfig


class TestConstants:
    def test_grid_size(self):
        assert GRID_SIZE == 78

    def test_num_threads(self):
        assert NUM_THREADS == 256

    def test_voxel_count(self):
        assert GRID_VOXEL_COUNT == 78**3

    def test_planck_length_order_of_magnitude(self):
        assert 1e-36 < PLANCK_LENGTH < 1e-34

    def test_observable_threshold(self):
        assert OBSERVABLE_THRESHOLD_M == pytest.approx(4e-7)

    def test_golden_ratio(self):
        assert GOLDEN_RATIO == pytest.approx(1.6180339887, rel=1e-8)


class TestSphericalCartesianRoundTrip:
    def test_origin(self):
        x, y, z = spherical_to_cartesian(0.0, 0.0, 0.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(0.0)

    def test_north_pole(self):
        x, y, z = spherical_to_cartesian(1.0, 0.0, 0.0)
        assert x == pytest.approx(0.0, abs=1e-15)
        assert y == pytest.approx(0.0, abs=1e-15)
        assert z == pytest.approx(1.0)

    def test_south_pole(self):
        x, y, z = spherical_to_cartesian(1.0, np.pi, 0.0)
        assert x == pytest.approx(0.0, abs=1e-15)
        assert y == pytest.approx(0.0, abs=1e-15)
        assert z == pytest.approx(-1.0)

    def test_equator_x(self):
        x, y, z = spherical_to_cartesian(1.0, np.pi / 2, 0.0)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0, abs=1e-15)
        assert z == pytest.approx(0.0, abs=1e-15)

    def test_round_trip(self):
        r_in, theta_in, phi_in = 2.5, 1.2, 3.7
        x, y, z = spherical_to_cartesian(r_in, theta_in, phi_in)
        r_out, theta_out, phi_out = cartesian_to_spherical(x, y, z)
        assert r_out == pytest.approx(r_in)
        assert theta_out == pytest.approx(theta_in)
        assert phi_out == pytest.approx(phi_in)

    def test_round_trip_array(self):
        r_in = np.array([1.0, 2.0, 3.0])
        theta_in = np.array([0.5, 1.0, 2.5])
        phi_in = np.array([0.3, 2.0, 5.0])
        x, y, z = spherical_to_cartesian(r_in, theta_in, phi_in)
        r_out, theta_out, phi_out = cartesian_to_spherical(x, y, z)
        np.testing.assert_allclose(r_out, r_in)
        np.testing.assert_allclose(theta_out, theta_in)
        np.testing.assert_allclose(phi_out, phi_in)


class TestUniformSpherePoints:
    def test_returns_correct_shape(self):
        pts = uniform_sphere_points(256)
        assert pts.shape == (256, 3)

    def test_all_unit_vectors(self):
        pts = uniform_sphere_points(256)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_small_n(self):
        pts = uniform_sphere_points(4)
        assert pts.shape == (4, 3)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_roughly_uniform_distribution(self):
        pts = uniform_sphere_points(1000)
        # Check that z values span nearly [-1, 1]
        assert pts[:, 2].min() < -0.95
        assert pts[:, 2].max() > 0.95
        # Check that mean is near origin
        mean = pts.mean(axis=0)
        np.testing.assert_allclose(mean, 0.0, atol=0.1)


class TestAngularGap:
    def test_parallel_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert angular_gap(v, v) == pytest.approx(0.0, abs=1e-14)

    def test_perpendicular_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert angular_gap(v1, v2) == pytest.approx(np.pi / 2)

    def test_antiparallel_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert angular_gap(v1, v2) == pytest.approx(np.pi)

    def test_45_degrees(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 0.0])
        assert angular_gap(v1, v2) == pytest.approx(np.pi / 4)


class TestNormalize:
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(normalize(v), v)

    def test_normalize_arbitrary(self):
        v = np.array([3.0, 4.0, 0.0])
        result = normalize(v)
        assert np.linalg.norm(result) == pytest.approx(1.0)
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0])

    def test_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


class TestPairwiseDistances:
    def test_identity(self):
        vectors = np.eye(3)
        dists = pairwise_angular_distances(vectors)
        assert dists.shape == (3, 3)
        # Diagonal should be 0
        np.testing.assert_allclose(np.diag(dists), 0.0, atol=1e-14)
        # Off-diagonal should be pi/2
        assert dists[0, 1] == pytest.approx(np.pi / 2)

    def test_symmetric(self):
        pts = uniform_sphere_points(10)
        dists = pairwise_angular_distances(pts)
        np.testing.assert_allclose(dists, dists.T)


class TestNearestNeighborGaps:
    def test_axis_vectors(self):
        vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        gaps = nearest_neighbor_gaps(vectors)
        # Each axis vector is pi/2 from the other two
        np.testing.assert_allclose(gaps, np.pi / 2)

    def test_returns_correct_shape(self):
        pts = uniform_sphere_points(256)
        gaps = nearest_neighbor_gaps(pts)
        assert gaps.shape == (256,)
        assert np.all(gaps > 0)


class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.grid_size == 78
        assert cfg.num_threads == 256
        assert cfg.resonance_freq == 78.0
        assert cfg.timesteps == 1000
