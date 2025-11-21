import pytest
from pytest import approx
import numpy as np
from app.streetGeometry import (
    depthFromAreaStreet,
    psiFromAreaStreet,
    psiPrimeFromAreaStreet,
)


class TestStreetGeometry:
    """Test suite for street geometry functions."""

    @pytest.fixture
    def street_params(self):
        """Standard street parameters for testing (in meters)."""
        ftToM = 0.3048
        return {
            "T_curb": 8 * ftToM,
            "T_crown": 15 * ftToM,
            "H_curb": 1 * ftToM,
            "S_back": 0.02,
            "Sx": 0.02,
        }

    @pytest.fixture
    def wide_street_params(self):
        """Wide street parameters."""
        ftToM = 0.3048
        return {
            "T_curb": 10 * ftToM,
            "T_crown": 20 * ftToM,
            "H_curb": 1.5 * ftToM,
            "S_back": 0.02,
            "Sx": 0.02,
        }

    @pytest.fixture
    def steep_street_params(self):
        """Steep cross-slope street parameters."""
        ftToM = 0.3048
        return {
            "T_curb": 8 * ftToM,
            "T_crown": 15 * ftToM,
            "H_curb": 1 * ftToM,
            "S_back": 0.04,
            "Sx": 0.04,
        }

    @pytest.fixture
    def area_boundaries(self, street_params):
        """Calculate area boundaries for different water level regions."""
        ps = street_params
        belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
        betweenCrownAndCurbArea = belowCrownArea + (
            ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"])
        )
        onSidewalkArea = (
            betweenCrownAndCurbArea
            + 0.5 * ps["T_curb"] * (ps["H_curb"] + ps["S_back"] * ps["T_curb"])
            + ps["T_crown"] * (ps["S_back"] * ps["T_curb"])
        )
        return {
            "belowCrown": belowCrownArea,
            "betweenCrownAndCurb": betweenCrownAndCurbArea,
            "onSidewalk": onSidewalkArea,
        }

    # Tests for depthFromAreaStreet
    def test_depth_from_area_zero(self, street_params):
        """Test depth when area is zero."""
        depth = depthFromAreaStreet(0.0, street_params)
        assert depth == approx(0.0)

    def test_depth_from_area_below_crown(self, street_params, area_boundaries):
        """Test depth calculation when water is below street crown."""
        ps = street_params
        # Test at 50% of below crown area
        A_test = 0.5 * area_boundaries["belowCrown"]
        depth = depthFromAreaStreet(A_test, ps)
        
        # Verify depth is positive and reasonable
        assert depth > 0
        assert depth < ps["Sx"] * ps["T_crown"]
        
        # Verify depth calculation formula
        expected_depth = np.sqrt(2 * ps["Sx"] * A_test)
        assert depth == approx(expected_depth, rel=1e-6)

    def test_depth_at_crown_boundary(self, street_params, area_boundaries):
        """Test depth at exactly the crown height (boundary)."""
        ps = street_params
        A_crown = area_boundaries["belowCrown"]
        depth = depthFromAreaStreet(A_crown, ps)
        
        # Depth should equal crown height
        expected_crown_depth = ps["Sx"] * ps["T_crown"]
        assert depth == approx(expected_crown_depth, rel=1e-3)

    def test_depth_between_crown_and_curb(self, street_params, area_boundaries):
        """Test depth when water is between crown and curb."""
        ps = street_params
        # Test at midpoint between crown and curb
        A_below = area_boundaries["belowCrown"]
        A_upper = area_boundaries["betweenCrownAndCurb"]
        A_test = 0.5 * (A_below + A_upper)
        
        depth = depthFromAreaStreet(A_test, ps)
        
        # Depth should be between crown height and curb height
        assert ps["Sx"] * ps["T_crown"] < depth < ps["H_curb"]

    def test_depth_at_curb_boundary(self, street_params, area_boundaries):
        """Test depth at exactly the curb height (boundary)."""
        ps = street_params
        A_curb = area_boundaries["betweenCrownAndCurb"]
        depth = depthFromAreaStreet(A_curb, ps)
        
        # Depth should equal curb height
        assert depth == approx(ps["H_curb"], rel=1e-3)

    def test_depth_on_sidewalk(self, street_params, area_boundaries):
        """Test depth when water is on sidewalk (above curb)."""
        ps = street_params
        # Test at midpoint on sidewalk
        A_curb = area_boundaries["betweenCrownAndCurb"]
        A_max = area_boundaries["onSidewalk"]
        A_test = 0.5 * (A_curb + A_max)
        
        depth = depthFromAreaStreet(A_test, ps)
        
        # Depth should be above curb height
        assert depth > ps["H_curb"]

    def test_depth_monotonicity(self, street_params, area_boundaries):
        """Test that depth increases monotonically with area."""
        A_max = area_boundaries["onSidewalk"]
        areas = np.linspace(0.01 * A_max, 0.99 * A_max, 30)
        depths = [depthFromAreaStreet(A, street_params) for A in areas]
        
        # Check monotonicity
        for i in range(len(depths) - 1):
            assert depths[i] < depths[i + 1] + 4e-2, f"Depth not monotonic at index {i}"

    def test_depth_continuity_at_boundaries(self, street_params, area_boundaries):
        """Test that depth is continuous at region boundaries."""
        ps = street_params
        epsilon = 1e-6
        
        # Test continuity at crown boundary
        A_crown = area_boundaries["belowCrown"]
        depth_below = depthFromAreaStreet(A_crown - epsilon, ps)
        depth_above = depthFromAreaStreet(A_crown + epsilon, ps)
        assert depth_below == approx(depth_above, rel=1e-2)
        
        # Test continuity at curb boundary
        A_curb = area_boundaries["betweenCrownAndCurb"]
        depth_below = depthFromAreaStreet(A_curb - epsilon, ps)
        depth_above = depthFromAreaStreet(A_curb + epsilon, ps)
        assert depth_below == approx(depth_above, rel=1e-2)

    def test_depth_different_street_sizes(self, street_params, wide_street_params):
        """Test depth calculation for different street sizes."""
        # Calculate relative area for both streets
        A_ratio = 0.5
        
        # Standard street
        ps1 = street_params
        A_max1 = 0.5 * ps1["T_crown"] * ps1["Sx"] * ps1["T_crown"]
        A_max1 += ps1["T_crown"] * (ps1["H_curb"] - ps1["Sx"] * ps1["T_crown"])
        depth1 = depthFromAreaStreet(A_ratio * A_max1, ps1)
        
        # Wide street
        ps2 = wide_street_params
        A_max2 = 0.5 * ps2["T_crown"] * ps2["Sx"] * ps2["T_crown"]
        A_max2 += ps2["T_crown"] * (ps2["H_curb"] - ps2["Sx"] * ps2["T_crown"])
        depth2 = depthFromAreaStreet(A_ratio * A_max2, ps2)
        
        # Both should be positive
        assert depth1 > 0
        assert depth2 > 0

    # Tests for psiFromAreaStreet
    def test_psi_from_area_zero(self, street_params):
        """Test psi when area is zero."""
        psi = psiFromAreaStreet(0.0, street_params)
        assert psi == approx(0.0)

    def test_psi_from_area_below_crown(self, street_params, area_boundaries):
        """Test psi calculation when water is below crown."""
        ps = street_params
        A_test = 0.5 * area_boundaries["belowCrown"]
        psi = psiFromAreaStreet(A_test, ps)
        
        # Verify psi is positive
        assert psi > 0
        
        # Verify psi calculation formula
        c = np.sqrt(2 * ps["Sx"]) * (1 + np.sqrt(1 + np.power(ps["Sx"], -2)))
        expected_psi = np.power(A_test, 4 / 3) * np.power(c, -2 / 3)
        assert psi == approx(expected_psi, rel=1e-6)

    def test_psi_from_area_between_crown_and_curb(self, street_params, area_boundaries):
        """Test psi when water is between crown and curb."""
        ps = street_params
        A_below = area_boundaries["belowCrown"]
        A_upper = area_boundaries["betweenCrownAndCurb"]
        A_test = 0.5 * (A_below + A_upper)
        
        psi = psiFromAreaStreet(A_test, ps)
        
        # Verify psi is positive
        assert psi > 0

    def test_psi_from_area_on_sidewalk(self, street_params, area_boundaries):
        """Test psi when water is on sidewalk."""
        ps = street_params
        A_curb = area_boundaries["betweenCrownAndCurb"]
        A_max = area_boundaries["onSidewalk"]
        A_test = 0.5 * (A_curb + A_max)
        
        psi = psiFromAreaStreet(A_test, ps)
        
        # Verify psi is positive
        assert psi > 0
        
        # Verify psi calculation formula for sidewalk region
        pStreet = (
            ps["Sx"] * ps["T_crown"] * np.sqrt(np.power(ps["Sx"], -1) + 1)
            + ps["H_curb"]
        )
        expected_psi = A_test * A_test * np.power(pStreet + ps["T_curb"], -1)
        assert psi == approx(expected_psi, rel=1e-6)

    def test_psi_monotonicity(self, street_params, area_boundaries):
        """Test that psi increases monotonically with area."""
        A_max = area_boundaries["onSidewalk"]
        areas = np.linspace(0.01 * A_max, 0.99 * A_max, 30)
        psis = [psiFromAreaStreet(A, street_params) for A in areas]
        
        # Check monotonicity
        for i in range(len(psis) - 1):
            assert psis[i] < psis[i + 1] + 5e-2, f"Psi not monotonic at index {i}, areas before: {areas[i]}, after: {areas[i+1]}"

    def test_psi_continuity_at_boundaries(self, street_params, area_boundaries):
        """Test that psi is continuous at region boundaries."""
        ps = street_params
        epsilon = 1e-6
        
        # Test continuity at crown boundary
        A_crown = area_boundaries["belowCrown"]
        psi_below = psiFromAreaStreet(A_crown - epsilon, ps)
        psi_above = psiFromAreaStreet(A_crown + epsilon, ps)
        assert psi_below == approx(psi_above, abs=5e-2)
        
        # Test continuity at curb boundary
        A_curb = area_boundaries["betweenCrownAndCurb"]
        psi_below = psiFromAreaStreet(A_curb - epsilon, ps)
        psi_above = psiFromAreaStreet(A_curb + epsilon, ps)
        assert psi_below == approx(psi_above, abs=5e-2)

    # Tests for psiPrimeFromAreaStreet
    def test_psi_prime_positive(self, street_params, area_boundaries):
        """Test that psi prime is positive for all valid areas."""
        A_max = area_boundaries["onSidewalk"]
        areas = np.linspace(0.01 * A_max, 0.99 * A_max, 30)
        
        for A in areas:
            psiPrime = psiPrimeFromAreaStreet(A, street_params)
            assert psiPrime > 0, f"psi prime not positive for A={A}"

    def test_psi_prime_below_crown(self, street_params, area_boundaries):
        """Test psi prime calculation when water is below crown."""
        ps = street_params
        A_test = 0.5 * area_boundaries["belowCrown"]
        psiPrime = psiPrimeFromAreaStreet(A_test, ps)
        
        # Verify psi prime is positive
        assert psiPrime > 0
        
        # Verify formula
        c = np.sqrt(2 * ps["Sx"]) * (1 + np.sqrt(1 + np.power(ps["Sx"], -2)))
        expected_psiPrime = (4 / 3) * np.power(A_test, 1 / 3) * np.power(c, -2 / 3)
        assert psiPrime == approx(expected_psiPrime, rel=1e-6)

    def test_psi_prime_between_crown_and_curb(self, street_params, area_boundaries):
        """Test psi prime when water is between crown and curb."""
        ps = street_params
        A_below = area_boundaries["belowCrown"]
        A_upper = area_boundaries["betweenCrownAndCurb"]
        A_test = 0.5 * (A_below + A_upper)
        
        psiPrime = psiPrimeFromAreaStreet(A_test, ps)
        
        # Verify psi prime is positive
        assert psiPrime > 0

    def test_psi_prime_on_sidewalk(self, street_params, area_boundaries):
        """Test psi prime when water is on sidewalk."""
        ps = street_params
        A_curb = area_boundaries["betweenCrownAndCurb"]
        A_max = area_boundaries["onSidewalk"]
        A_test = 0.5 * (A_curb + A_max)
        
        psiPrime = psiPrimeFromAreaStreet(A_test, ps)
        
        # Verify psi prime is positive
        assert psiPrime > 0
        
        # Verify formula for sidewalk region
        pStreet = (
            ps["Sx"] * ps["T_crown"] * np.sqrt(np.power(ps["Sx"], -1) + 1)
            + ps["H_curb"]
        )
        expected_psiPrime = 2 * A_test * np.power(pStreet + ps["T_curb"], -1)
        assert psiPrime == approx(expected_psiPrime, rel=1e-6)

    def test_psi_prime_consistency_with_psi(self, street_params, area_boundaries):
        """Test that psi prime is approximately the derivative of psi."""
        A_max = area_boundaries["onSidewalk"]
        A = 0.5 * A_max
        delta_A = 0.001 * A_max
        
        psi1 = psiFromAreaStreet(A, street_params)
        psi2 = psiFromAreaStreet(A + delta_A, street_params)
        numerical_derivative = (psi2 - psi1) / delta_A
        
        psiPrime = psiPrimeFromAreaStreet(A, street_params)
        
        # Should be roughly equal
        assert psiPrime == approx(numerical_derivative, rel=0.1)

    def test_psi_prime_continuity_at_boundaries(self, street_params, area_boundaries):
        """Test that psi prime has reasonable continuity at boundaries."""
        ps = street_params
        epsilon = 1e-5
        
        # Test at crown boundary
        A_crown = area_boundaries["belowCrown"]
        psiPrime_below = psiPrimeFromAreaStreet(A_crown - epsilon, ps)
        psiPrime_above = psiPrimeFromAreaStreet(A_crown + epsilon, ps)
        # Should be same order of magnitude
        assert 0.1 < psiPrime_below / psiPrime_above < 10
        
        # Test at curb boundary
        A_curb = area_boundaries["betweenCrownAndCurb"]
        psiPrime_below = psiPrimeFromAreaStreet(A_curb - epsilon, ps)
        psiPrime_above = psiPrimeFromAreaStreet(A_curb + epsilon, ps)
        # Should be same order of magnitude
        assert 0.1 < psiPrime_below / psiPrime_above < 10

    # Integration tests
    def test_consistent_geometry_calculations(self, street_params, area_boundaries):
        """Test that all geometric functions are consistent with each other."""
        A_max = area_boundaries["onSidewalk"]
        A_test = 0.6 * A_max
        
        # Get all geometric properties
        depth = depthFromAreaStreet(A_test, street_params)
        psi = psiFromAreaStreet(A_test, street_params)
        psi_prime = psiPrimeFromAreaStreet(A_test, street_params)
        
        # All should be positive
        assert depth > 0
        assert psi > 0
        assert psi_prime > 0

    def test_all_regions_with_steep_slope(self, steep_street_params):
        """Test all functions work with steep cross slopes."""
        ps = steep_street_params
        belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
        betweenCrownAndCurbArea = belowCrownArea + (
            ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"])
        )
        
        # Test in each region
        areas = [
            0.5 * belowCrownArea,  # Below crown
            belowCrownArea + 0.5 * (betweenCrownAndCurbArea - belowCrownArea),  # Between
            betweenCrownAndCurbArea + 0.01,  # On sidewalk
        ]
        
        for A in areas:
            depth = depthFromAreaStreet(A, ps)
            psi = psiFromAreaStreet(A, ps)
            psi_prime = psiPrimeFromAreaStreet(A, ps)
            
            assert depth > 0
            assert psi > 0
            assert psi_prime > 0

    def test_numerical_stability_small_areas(self, street_params):
        """Test numerical stability with very small areas."""
        A_tiny = 1e-8
        
        depth = depthFromAreaStreet(A_tiny, street_params)
        psi = psiFromAreaStreet(A_tiny, street_params)
        psi_prime = psiPrimeFromAreaStreet(A_tiny, street_params)
        
        assert np.isfinite(depth)
        assert np.isfinite(psi)
        assert np.isfinite(psi_prime)
        assert depth >= 0
        assert psi >= 0
        assert psi_prime > 0

    def test_all_functions_at_multiple_areas(self, street_params, area_boundaries):
        """Comprehensive test of all functions at various areas."""
        A_max = area_boundaries["onSidewalk"]
        area_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        for ratio in area_ratios:
            A = ratio * A_max
            
            # All functions should complete without error
            depth = depthFromAreaStreet(A, street_params)
            psi = psiFromAreaStreet(A, street_params)
            psi_prime = psiPrimeFromAreaStreet(A, street_params)
            
            # Basic sanity checks
            assert depth > 0
            assert psi > 0
            assert psi_prime > 0

    def test_depth_matches_expected_at_key_points(self, street_params):
        """Test depth matches expected values at key geometric points."""
        ps = street_params
        
        # At crown (end of first region)
        A_crown = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
        depth_crown = depthFromAreaStreet(A_crown, ps)
        expected_crown_depth = ps["Sx"] * ps["T_crown"]
        assert depth_crown == approx(expected_crown_depth, rel=1e-3)
        
        # At curb (end of second region)
        A_curb = A_crown + ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"])
        depth_curb = depthFromAreaStreet(A_curb, ps)
        assert depth_curb == approx(ps["H_curb"], rel=1e-3)

    def test_different_parameter_combinations(self):
        """Test with various realistic parameter combinations."""
        test_params = [
            {"T_curb": 2.4, "T_crown": 4.5, "H_curb": 0.15, "S_back": 0.02, "Sx": 0.02},
            {"T_curb": 3.0, "T_crown": 6.0, "H_curb": 0.20, "S_back": 0.03, "Sx": 0.03},
            {"T_curb": 2.0, "T_crown": 5.0, "H_curb": 0.10, "S_back": 0.015, "Sx": 0.015},
        ]
        
        for ps in test_params:
            # Calculate a test area in the middle region
            belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
            betweenCrownAndCurbArea = belowCrownArea + (
                ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"])
            )
            A_test = 0.5 * (belowCrownArea + betweenCrownAndCurbArea)
            
            # All functions should work
            depth = depthFromAreaStreet(A_test, ps)
            psi = psiFromAreaStreet(A_test, ps)
            psi_prime = psiPrimeFromAreaStreet(A_test, ps)
            
            assert depth > 0
            assert psi > 0
            assert psi_prime > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
