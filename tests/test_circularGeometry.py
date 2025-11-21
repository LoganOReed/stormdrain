import pytest
import numpy as np
from app.circularGeometry import (
    _angleFromArea,
    depthFromAreaCircle,
    psiFromAreaCircle,
    areaFromPsiCircle,
    psiPrimeFromAreaCircle,
    hydraulicRadiusFromAreaCircle,
)


class TestCircularGeometry:
    """Test suite for circular geometry functions."""

    @pytest.fixture
    def pipe_params(self):
        """Standard pipe parameters for testing."""
        return {"yFull": 0.5}

    @pytest.fixture
    def large_pipe_params(self):
        """Large pipe parameters."""
        return {"yFull": 1.0}

    @pytest.fixture
    def small_pipe_params(self):
        """Small pipe parameters."""
        return {"yFull": 0.1}

    # Tests for _angleFromArea
    def test_angle_from_area_empty_pipe(self, pipe_params):
        """Test angle calculation when area is zero."""
        A = 0.0
        theta = _angleFromArea(A, pipe_params["yFull"])
        assert np.isclose(theta, 0.0, atol=1e-5)

    def test_angle_from_area_full_pipe(self, pipe_params):
        """Test angle calculation when pipe is full (A = Afull)."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        theta = _angleFromArea(Afull, pipe_params["yFull"])
        # Full pipe should have theta = 2*pi
        assert np.isclose(theta, 2 * np.pi, atol=1e-5)

    def test_angle_from_area_half_full(self, pipe_params):
        """Test angle calculation when pipe is half full."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_half = 0.5 * Afull
        theta = _angleFromArea(A_half, pipe_params["yFull"])
        # Half area corresponds to theta ≈ π
        assert 2.5 < theta < 3.5  # Approximate range for half area

    def test_angle_from_area_quarter_full(self, pipe_params):
        """Test angle calculation when pipe is quarter full."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_quarter = 0.25 * Afull
        theta = _angleFromArea(A_quarter, pipe_params["yFull"])
        assert theta > 0 and theta < np.pi

    def test_angle_from_area_increases_with_area(self, pipe_params):
        """Test that angle increases monotonically with area."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        areas = np.linspace(0.01 * Afull, 0.99 * Afull, 10)
        angles = [_angleFromArea(A, pipe_params["yFull"]) for A in areas]

        for i in range(len(angles) - 1):
            assert angles[i] < angles[i + 1]

    # Tests for depthFromAreaCircle
    def test_depth_from_area_empty(self, pipe_params):
        """Test depth when area is zero."""
        depth = depthFromAreaCircle(0.0, pipe_params)
        assert np.isclose(depth, 0.0, atol=1e-6)

    def test_depth_from_area_full(self, pipe_params):
        """Test depth when pipe is full."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        depth = depthFromAreaCircle(Afull, pipe_params)
        assert np.isclose(depth, Yfull, atol=1e-3)

    def test_depth_from_area_half_full(self, pipe_params):
        """Test depth when pipe is approximately half full by area."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_half = 0.5 * Afull
        depth = depthFromAreaCircle(A_half, pipe_params)
        # Depth at half area should be less than Yfull/2 due to circular geometry
        assert 0.3 * Yfull < depth < 0.6 * Yfull

    def test_depth_from_area_small_area(self, pipe_params):
        """Test depth for small area (< 4% full) using direct calculation."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_small = 0.02 * Afull  # 2% full
        depth = depthFromAreaCircle(A_small, pipe_params)
        assert depth > 0
        assert depth < 0.1 * Yfull

    def test_depth_from_area_large_area(self, pipe_params):
        """Test depth for large area (> 4% full) using interpolation."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_large = 0.75 * Afull
        depth = depthFromAreaCircle(A_large, pipe_params)
        assert 0.5 * Yfull < depth < Yfull

    def test_depth_monotonicity(self, pipe_params):
        """Test that depth increases monotonically with area."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        areas = np.linspace(0.01 * Afull, 0.99 * Afull, 20)
        depths = [depthFromAreaCircle(A, pipe_params) for A in areas]

        # Check monotonicity
        for i in range(len(depths) - 1):
            assert depths[i] < depths[i + 1], f"Depth not monotonic at index {i}"

    def test_depth_different_pipe_sizes(self, small_pipe_params, large_pipe_params):
        """Test depth calculation scales with pipe size."""
        # Same relative area, different pipe sizes
        A_ratio = 0.5

        Yfull_small = small_pipe_params["yFull"]
        Afull_small = 0.7854 * Yfull_small * Yfull_small
        depth_small = depthFromAreaCircle(A_ratio * Afull_small, small_pipe_params)

        Yfull_large = large_pipe_params["yFull"]
        Afull_large = 0.7854 * Yfull_large * Yfull_large
        depth_large = depthFromAreaCircle(A_ratio * Afull_large, large_pipe_params)

        # Relative depths should be similar
        assert np.isclose(
            depth_small / Yfull_small, depth_large / Yfull_large, rtol=0.01
        )

    def test_depth_boundary_between_methods(self, pipe_params):
        """Test depth calculation near the 4% boundary where method switches."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull

        # Just below 4%
        A_below = 0.039 * Afull
        depth_below = depthFromAreaCircle(A_below, pipe_params)

        # Just above 4%
        A_above = 0.041 * Afull
        depth_above = depthFromAreaCircle(A_above, pipe_params)

        # Should be continuous
        assert depth_below < depth_above
        assert np.isclose(depth_below, depth_above, rtol=0.1)

    # Tests for psiFromAreaCircle
    def test_psi_from_area_zero(self, pipe_params):
        """Test section factor when area is zero."""
        psi = psiFromAreaCircle(0.0, pipe_params)
        assert np.isclose(psi, 0.0, atol=1e-6)

    def test_psi_from_area_full(self, pipe_params):
        """Test section factor when pipe is full."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        Rfull = 0.25 * Yfull
        PsiFull = Afull * np.power(Rfull, 2 / 3)

        psi = psiFromAreaCircle(Afull, pipe_params)
        assert np.isclose(psi, PsiFull, rtol=0.01)

    def test_psi_monotonicity(self, pipe_params):
        """Test that psi increases monotonically with area."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        areas = np.linspace(0.01 * Afull, 0.99 * Afull, 20)
        psis = [psiFromAreaCircle(A, pipe_params) for A in areas]

        # Check monotonicity
        for i in range(len(psis) - 1):
            assert psis[i] < psis[i + 1], f"Psi not monotonic at index {i}"

    def test_psi_small_area_branch(self, pipe_params):
        """Test psi calculation for small area (< 4% full)."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_small = 0.03 * Afull
        psi = psiFromAreaCircle(A_small, pipe_params)
        assert psi > 0
        assert psi < psiFromAreaCircle(0.05 * Afull, pipe_params)

    def test_psi_large_area_branch(self, pipe_params):
        """Test psi calculation for large area (> 4% full)."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_large = 0.5 * Afull
        psi = psiFromAreaCircle(A_large, pipe_params)
        assert psi > 0

    def test_psi_boundary_between_methods(self, pipe_params):
        """Test psi calculation near the 4% boundary."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull

        A_below = 0.039 * Afull
        psi_below = psiFromAreaCircle(A_below, pipe_params)

        A_above = 0.041 * Afull
        psi_above = psiFromAreaCircle(A_above, pipe_params)

        assert psi_below < psi_above
        assert np.isclose(psi_below, psi_above, rtol=0.1)

    # Tests for areaFromPsiCircle (inverse function)
    def test_area_from_psi_roundtrip(self, pipe_params):
        """Test that areaFromPsi is inverse of psiFromArea."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        # Test across full range including extreme values
        test_areas = [
            0.02 * Afull,
            0.1 * Afull,
            0.3 * Afull,
            0.5 * Afull,
            0.7 * Afull,
            0.9 * Afull,
            0.95 * Afull,
        ]

        for A_original in test_areas:
            psi = psiFromAreaCircle(A_original, pipe_params)
            A_recovered = areaFromPsiCircle(psi, pipe_params)
            assert np.isclose(A_original, A_recovered, rtol=0.03)

    def test_area_from_psi_zero(self, pipe_params):
        """Test area from psi when psi is zero."""
        A = areaFromPsiCircle(0.0, pipe_params)
        assert np.isclose(A, 0.0, atol=1e-6)

    def test_area_from_psi_full(self, pipe_params):
        """Test area from psi when pipe is full."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        Rfull = 0.25 * Yfull
        PsiFull = Afull * np.power(Rfull, 2 / 3)

        A = areaFromPsiCircle(PsiFull, pipe_params)
        assert np.isclose(A, Afull, rtol=0.01)

    # Tests for psiPrimeFromAreaCircle
    def test_psi_prime_positive(self, pipe_params):
        """Test that psi prime is positive for all valid areas."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        areas = np.linspace(0.01 * Afull, 0.99 * Afull, 20)

        for A in areas:
            psiPrime = psiPrimeFromAreaCircle(A, pipe_params)
            assert psiPrime > 0, f"psi prime not positive for A={A}"

    def test_psi_prime_small_area_branch(self, pipe_params):
        """Test psi prime calculation for small area (< 4% full)."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_small = 0.02 * Afull
        psiPrime = psiPrimeFromAreaCircle(A_small, pipe_params)
        assert psiPrime > 0

    def test_psi_prime_large_area_branch(self, pipe_params):
        """Test psi prime calculation for large area (> 4% full)."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_large = 0.5 * Afull
        psiPrime = psiPrimeFromAreaCircle(A_large, pipe_params)
        assert psiPrime > 0

    def test_psi_prime_consistency_with_psi(self, pipe_params):
        """Test that psi prime is approximately the derivative of psi."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A = 0.5 * Afull
        delta_A = 0.001 * Afull

        psi1 = psiFromAreaCircle(A, pipe_params)
        psi2 = psiFromAreaCircle(A + delta_A, pipe_params)
        numerical_derivative = (psi2 - psi1) / delta_A

        psiPrime = psiPrimeFromAreaCircle(A, pipe_params)

        # Should be roughly equal
        assert np.isclose(psiPrime, numerical_derivative, rtol=0.1)

    def test_psi_prime_different_pipe_sizes(self, small_pipe_params, large_pipe_params):
        """Test psi prime scales appropriately with pipe size."""
        A_ratio = 0.5

        Yfull_small = small_pipe_params["yFull"]
        Afull_small = 0.7854 * Yfull_small * Yfull_small
        psiPrime_small = psiPrimeFromAreaCircle(
            A_ratio * Afull_small, small_pipe_params
        )

        Yfull_large = large_pipe_params["yFull"]
        Afull_large = 0.7854 * Yfull_large * Yfull_large
        psiPrime_large = psiPrimeFromAreaCircle(
            A_ratio * Afull_large, large_pipe_params
        )

        # Both should be positive
        assert psiPrime_small > 0
        assert psiPrime_large > 0

    # Tests for hydraulicRadiusFromAreaCircle
    def test_hydraulic_radius_zero_area(self, pipe_params):
        """Test hydraulic radius when area is very small."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_tiny = 0.001 * Afull
        r = hydraulicRadiusFromAreaCircle(A_tiny, pipe_params)
        assert r >= 0
        assert r < 0.01 * Yfull

    def test_hydraulic_radius_full_pipe(self, pipe_params):
        """Test hydraulic radius when pipe is full."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        Rfull = 0.25 * Yfull

        r = hydraulicRadiusFromAreaCircle(Afull, pipe_params)
        assert np.isclose(r, Rfull, rtol=0.01)

    def test_hydraulic_radius_half_area(self, pipe_params):
        """Test hydraulic radius at half area."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        Rfull = 0.25 * Yfull
        A_half = 0.5 * Afull

        r = hydraulicRadiusFromAreaCircle(A_half, pipe_params)
        assert 0 < r <= Rfull

    def test_hydraulic_radius_small_area_branch(self, pipe_params):
        """Test hydraulic radius for small area (< 4% full)."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_small = 0.03 * Afull
        r = hydraulicRadiusFromAreaCircle(A_small, pipe_params)
        assert r > 0
        assert r < 0.25 * Yfull

    def test_hydraulic_radius_large_area_branch(self, pipe_params):
        """Test hydraulic radius for large area (> 4% full)."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_large = 0.6 * Afull
        r = hydraulicRadiusFromAreaCircle(A_large, pipe_params)
        assert r > 0

    def test_hydraulic_radius_physical_bounds(self, pipe_params):
        """Test that hydraulic radius stays within physical bounds."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        Rfull = 0.25 * Yfull

        areas = np.linspace(0.01 * Afull, 0.99 * Afull, 20)
        for A in areas:
            r = hydraulicRadiusFromAreaCircle(A, pipe_params)
            # R should be between 0 and D/4 for circular pipes
            assert 0 <= r <= Rfull * 1.1, f"R out of bounds for A={A}"

    def test_hydraulic_radius_boundary_between_methods(self, pipe_params):
        """Test hydraulic radius near the 4% boundary."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull

        A_below = 0.039 * Afull
        r_below = hydraulicRadiusFromAreaCircle(A_below, pipe_params)

        A_above = 0.041 * Afull
        r_above = hydraulicRadiusFromAreaCircle(A_above, pipe_params)

        # Should be continuous
        assert r_below < r_above
        assert np.isclose(r_below, r_above, rtol=0.1)

    # Integration tests
    def test_consistent_geometry_calculations(self, pipe_params):
        """Test that all geometric functions are consistent with each other."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull
        A_test = 0.6 * Afull

        # Get all geometric properties
        depth = depthFromAreaCircle(A_test, pipe_params)
        psi = psiFromAreaCircle(A_test, pipe_params)
        r = hydraulicRadiusFromAreaCircle(A_test, pipe_params)

        # All should be positive and reasonable
        assert 0 < depth < Yfull
        assert psi > 0
        assert 0 < r < 0.25 * Yfull

    def test_multiple_pipe_sizes_consistency(self):
        """Test that functions work consistently across different pipe sizes."""
        pipe_sizes = [0.1, 0.5, 1.0, 2.5]
        A_ratio = 0.5

        relative_depths = []
        for Yfull in pipe_sizes:
            p = {"yFull": Yfull}
            Afull = 0.7854 * Yfull * Yfull
            depth = depthFromAreaCircle(A_ratio * Afull, p)
            relative_depths.append(depth / Yfull)

        # All relative depths should be very similar
        for i in range(len(relative_depths) - 1):
            assert np.isclose(relative_depths[i], relative_depths[i + 1], rtol=0.01)

    def test_numerical_stability_extreme_values(self, pipe_params):
        """Test numerical stability with very small and very large areas."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull

        # Very small area
        A_tiny = 1e-6 * Afull
        depth_tiny = depthFromAreaCircle(A_tiny, pipe_params)
        assert np.isfinite(depth_tiny)
        assert depth_tiny >= 0

        # Very large area (but not quite full)
        A_large = 0.9999 * Afull
        depth_large = depthFromAreaCircle(A_large, pipe_params)
        assert np.isfinite(depth_large)
        assert depth_large <= Yfull

    def test_all_functions_at_multiple_areas(self, pipe_params):
        """Comprehensive test of all functions at various area ratios."""
        Yfull = pipe_params["yFull"]
        Afull = 0.7854 * Yfull * Yfull

        area_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]

        for ratio in area_ratios:
            A = ratio * Afull

            # All functions should complete without error
            depth = depthFromAreaCircle(A, pipe_params)
            psi = psiFromAreaCircle(A, pipe_params)
            psi_prime = psiPrimeFromAreaCircle(A, pipe_params)
            r = hydraulicRadiusFromAreaCircle(A, pipe_params)

            # Basic sanity checks
            assert 0 < depth <= Yfull
            assert psi > 0
            assert 0 < r <= 0.25 * Yfull


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
