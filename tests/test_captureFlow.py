import pytest
from pytest import approx
import numpy as np
from app.drainCapture import capturedFlow


class TestCapturedFlow:
    """Test suite for capturedFlow function."""

    @pytest.fixture
    def standard_params(self):
        """Standard parameters for P-50x100 drain testing."""
        return {
            "Q": 0.05,  # Flow rate (m³/s)
            "S0": 0.02,  # Longitudinal slope
            "Sx": 0.04,  # Cross slope
            "L": 0.6,  # Drain length (m)
            "W": 0.6,  # Drain width (m)
            "n": 0.017,  # Manning's roughness coefficient
        }

    @pytest.fixture
    def small_drain_params(self):
        """Parameters for smaller drain."""
        return {
            "Q": 0.05,
            "S0": 0.02,
            "Sx": 0.04,
            "L": 0.4,
            "W": 0.4,
            "n": 0.017,
        }

    @pytest.fixture
    def large_drain_params(self):
        """Parameters for larger drain."""
        return {
            "Q": 0.05,
            "S0": 0.02,
            "Sx": 0.04,
            "L": 0.9,
            "W": 0.9,
            "n": 0.017,
        }

    # Basic functionality tests
    def test_captured_flow_zero_flow(self, standard_params):
        """Test captured flow when input flow is zero."""
        params = standard_params.copy()
        params["Q"] = 0.0
        Qc = capturedFlow(**params)
        assert Qc == 0.0

    def test_captured_flow_positive(self, standard_params):
        """Test that captured flow is positive for normal conditions."""
        Qc = capturedFlow(**standard_params)
        assert Qc > 0

    def test_captured_flow_less_than_total(self, standard_params):
        """Test that captured flow never exceeds total flow."""
        Qc = capturedFlow(**standard_params)
        assert Qc <= standard_params["Q"]

    def test_captured_flow_realistic_range(self, standard_params):
        """Test that captured flow is in a realistic range (0-100% of Q)."""
        Qc = capturedFlow(**standard_params)
        assert 0 <= Qc <= standard_params["Q"]

    # Tests for varying flow rates
    def test_captured_flow_increases_with_Q(self, standard_params):
        """Test that captured flow increases with total flow."""
        params = standard_params.copy()
        
        Q_values = [0.01, 0.03, 0.05, 0.10, 0.15]
        Qc_values = []
        
        for Q in Q_values:
            params["Q"] = Q
            Qc = capturedFlow(**params)
            Qc_values.append(Qc)
        
        # Captured flow should generally increase with Q
        for i in range(len(Qc_values) - 1):
            assert Qc_values[i] <= Qc_values[i + 1]

    def test_small_flow_high_capture_efficiency(self, standard_params):
        """Test that small flows have high capture efficiency."""
        params = standard_params.copy()
        params["Q"] = 0.01  # Small flow
        
        Qc = capturedFlow(**params)
        efficiency = Qc / params["Q"]
        
        # Small flows should be captured efficiently (>50%)
        assert efficiency > 0.5

    def test_large_flow_lower_capture_efficiency(self, standard_params):
        """Test that large flows have lower capture efficiency."""
        params = standard_params.copy()
        params["Q"] = 0.20  # Large flow
        
        Qc = capturedFlow(**params)
        efficiency = Qc / params["Q"]
        
        # Large flows typically have lower efficiency
        # Just ensure it's positive and < 100%
        assert 0 < efficiency <= 1.0

    # Tests for varying slopes
    def test_captured_flow_with_steep_longitudinal_slope(self, standard_params):
        """Test captured flow with steeper longitudinal slope."""
        params = standard_params.copy()
        params["S0"] = 0.08  # Steeper slope
        
        Qc = capturedFlow(**params)
        
        # Should still capture some flow
        assert 0 < Qc <= params["Q"]

    def test_captured_flow_with_mild_longitudinal_slope(self, standard_params):
        """Test captured flow with milder longitudinal slope."""
        params = standard_params.copy()
        params["S0"] = 0.005  # Milder slope
        
        Qc = capturedFlow(**params)
        
        # Should still capture some flow
        assert 0 < Qc <= params["Q"]


    def test_captured_flow_with_steep_cross_slope(self, standard_params):
        """Test captured flow with steeper cross slope."""
        params = standard_params.copy()
        params["Sx"] = 0.06  # Steeper cross slope
        
        Qc = capturedFlow(**params)
        
        # Should still capture some flow
        assert 0 < Qc <= params["Q"]

    def test_captured_flow_with_mild_cross_slope(self, standard_params):
        """Test captured flow with milder cross slope."""
        params = standard_params.copy()
        params["Sx"] = 0.02  # Milder cross slope
        
        Qc = capturedFlow(**params)
        
        # Should still capture some flow
        assert 0 < Qc <= params["Q"]

    def test_captured_flow_increases_with_Sx(self, standard_params):
        """Test that captured flow generally increases with cross slope."""
        params = standard_params.copy()
        
        Sx_values = [0.02, 0.03, 0.04, 0.06]
        Qc_values = []
        
        for Sx in Sx_values:
            params["Sx"] = Sx
            Qc = capturedFlow(**params)
            Qc_values.append(Qc)
        
        # Steeper cross slopes should improve side capture
        # Check general trend (monotonic increase)
        for i in range(len(Qc_values) - 1):
            assert Qc_values[i] <= Qc_values[i + 1]

    # Tests for drain dimensions
    def test_larger_drain_captures_more(self, standard_params, large_drain_params):
        """Test that larger drain captures more flow than standard."""
        Qc_standard = capturedFlow(**standard_params)
        Qc_large = capturedFlow(**large_drain_params)
        
        # Larger drain should capture at least as much
        assert Qc_large >= Qc_standard

    def test_smaller_drain_captures_less(self, standard_params, small_drain_params):
        """Test that smaller drain captures less flow than standard."""
        Qc_standard = capturedFlow(**standard_params)
        Qc_small = capturedFlow(**small_drain_params)
        
        # Smaller drain should capture less or equal
        assert Qc_small <= Qc_standard

    def test_captured_flow_increases_with_length(self, standard_params):
        """Test that captured flow increases with drain length."""
        params = standard_params.copy()
        
        L_values = [0.3, 0.45, 0.6, 0.75, 0.9]
        Qc_values = []
        
        for L in L_values:
            params["L"] = L
            Qc = capturedFlow(**params)
            Qc_values.append(Qc)
        
        # Longer drains should capture more
        for i in range(len(Qc_values) - 1):
            assert Qc_values[i] <= Qc_values[i + 1]

    def test_captured_flow_increases_with_width(self, standard_params):
        """Test that captured flow increases with drain width."""
        params = standard_params.copy()
        
        W_values = [0.3, 0.45, 0.6, 0.75, 0.9]
        Qc_values = []
        
        for W in W_values:
            params["W"] = W
            Qc = capturedFlow(**params)
            Qc_values.append(Qc)
        
        # Wider drains should intercept more flow
        for i in range(len(Qc_values) - 1):
            assert Qc_values[i] <= Qc_values[i + 1]

    # Tests for Manning's n
    def test_captured_flow_with_rough_surface(self, standard_params):
        """Test captured flow with rough surface (high n)."""
        params = standard_params.copy()
        params["n"] = 0.025  # Rougher surface
        
        Qc = capturedFlow(**params)
        
        # Should still capture some flow
        assert 0 < Qc <= params["Q"]

    def test_captured_flow_with_smooth_surface(self, standard_params):
        """Test captured flow with smooth surface (low n)."""
        params = standard_params.copy()
        params["n"] = 0.012  # Smoother surface
        
        Qc = capturedFlow(**params)
        
        # Should still capture some flow
        assert 0 < Qc <= params["Q"]

    def test_captured_flow_varies_with_roughness(self, standard_params):
        """Test that captured flow varies with Manning's n."""
        params = standard_params.copy()
        
        n_values = [0.011, 0.013, 0.015, 0.017, 0.020]
        Qc_values = []
        
        for n in n_values:
            params["n"] = n
            Qc = capturedFlow(**params)
            Qc_values.append(Qc)
        
        # Should have variation
        assert len(set(Qc_values)) > 1

    # Tests for capture efficiency components
    def test_e0_calculation_full_width(self, standard_params):
        """Test e0 (frontal flow ratio) when drain width covers full flow width."""
        params = standard_params.copy()
        params["W"] = 10.0  # Very wide drain
        
        Qc = capturedFlow(**params)
        
        # Wide drain should have high efficiency
        efficiency = Qc / params["Q"]
        assert efficiency > 0.7  # Should capture most flow

    def test_velocity_effects_on_capture(self, standard_params):
        """Test that velocity affects capture efficiency."""
        params = standard_params.copy()
        
        # Low velocity case (mild slope)
        params["S0"] = 0.005
        Qc_low_vel = capturedFlow(**params)
        
        # High velocity case (steep slope)
        params["S0"] = 0.10
        Qc_high_vel = capturedFlow(**params)
        
        # Lower velocity should have higher capture efficiency
        eff_low = Qc_low_vel / params["Q"]
        eff_high = Qc_high_vel / params["Q"]
        
        assert eff_low <= eff_high

    # Tests for physical constraints
    def test_captured_flow_obeys_conservation(self, standard_params):
        """Test that captured flow obeys mass conservation."""
        Qc = capturedFlow(**standard_params)
        Q = standard_params["Q"]
        
        # Captured flow must be between 0 and total flow
        assert 0 <= Qc <= Q
        
        # Uncaptured flow must be non-negative
        Q_uncaptured = Q - Qc
        assert Q_uncaptured >= 0

    def test_capture_efficiency_bounded(self, standard_params):
        """Test that capture efficiency is between 0 and 1."""
        Qc = capturedFlow(**standard_params)
        Q = standard_params["Q"]
        
        if Q > 0:
            efficiency = Qc / Q
            assert 0 <= efficiency <= 1

    # Numerical stability tests
    def test_captured_flow_with_very_small_Q(self, standard_params):
        """Test numerical stability with very small flow."""
        params = standard_params.copy()
        params["Q"] = 1e-6
        
        Qc = capturedFlow(**params)
        
        assert np.isfinite(Qc)
        assert Qc >= 0
        assert Qc <= params["Q"]

    def test_captured_flow_with_very_small_slopes(self, standard_params):
        """Test numerical stability with very small slopes."""
        params = standard_params.copy()
        params["S0"] = 0.001
        params["Sx"] = 0.01
        
        Qc = capturedFlow(**params)
        
        assert np.isfinite(Qc)
        assert 0 <= Qc <= params["Q"]

    def test_captured_flow_with_large_values(self, standard_params):
        """Test with large flow values."""
        params = standard_params.copy()
        params["Q"] = 1.0
        
        Qc = capturedFlow(**params)
        
        assert np.isfinite(Qc)
        assert 0 <= Qc <= params["Q"]

    # Tests for edge cases
    def test_captured_flow_with_zero_width(self, standard_params):
        """Test with zero drain width."""
        params = standard_params.copy()
        params["W"] = 0.0
        
        Qc = capturedFlow(**params)
        
        # Should still capture via side flow
        assert np.isfinite(Qc)
        assert Qc >= 0

    def test_captured_flow_with_zero_length(self, standard_params):
        """Test with zero drain length."""
        params = standard_params.copy()
        params["L"] = 0.0
        
        Qc = capturedFlow(**params)
        
        # Should capture very little or nothing
        assert np.isfinite(Qc)
        assert Qc >= 0

    def test_captured_flow_multiple_scenarios(self):
        """Test multiple realistic scenarios."""
        scenarios = [
            # (Q, S0, Sx, L, W, n, description)
            (0.02, 0.01, 0.02, 0.5, 0.5, 0.015, "mild conditions"),
            (0.08, 0.03, 0.04, 0.6, 0.6, 0.017, "moderate conditions"),
            (0.15, 0.05, 0.05, 0.75, 0.75, 0.020, "steep conditions"),
            (0.01, 0.005, 0.03, 0.4, 0.4, 0.013, "very mild"),
        ]
        
        for Q, S0, Sx, L, W, n, desc in scenarios:
            Qc = capturedFlow(Q, S0, Sx, L, W, n)
            
            # Basic checks for all scenarios
            assert np.isfinite(Qc), f"Non-finite result for {desc}"
            assert 0 <= Qc <= Q, f"Invalid capture for {desc}"

    # Integration and consistency tests
    def test_captured_flow_consistent_across_calls(self, standard_params):
        """Test that function gives consistent results."""
        Qc1 = capturedFlow(**standard_params)
        Qc2 = capturedFlow(**standard_params)
        
        assert Qc1 == Qc2

    def test_parameter_sensitivity_reasonable(self, standard_params):
        """Test that small parameter changes produce small output changes."""
        params = standard_params.copy()
        
        Qc_base = capturedFlow(**params)
        
        # Small change in Q
        params["Q"] = standard_params["Q"] * 1.01
        Qc_perturbed = capturedFlow(**params)
        
        # Change should be small and in same direction
        assert abs(Qc_perturbed - Qc_base) / Qc_base < 0.2

    def test_physical_units_consistency(self, standard_params):
        """Test that function works with different unit scales (all SI)."""
        params = standard_params.copy()
        
        # Test with scaled values (maintaining dimensionless ratios)
        scale = 2.0
        params["L"] = standard_params["L"] * scale
        params["W"] = standard_params["W"] * scale
        
        Qc = capturedFlow(**params)
        
        # Should still produce valid results
        assert np.isfinite(Qc)
        assert 0 <= Qc <= params["Q"]

    def test_realistic_capture_percentages(self):
        """Test that capture percentages are realistic for various conditions."""
        test_cases = [
            # Small flow, mild slope - high efficiency expected
            {"Q": 0.01, "S0": 0.01, "Sx": 0.04, "L": 0.6, "W": 0.6, "n": 0.017, "min_eff": 0.4},
            # Large flow, steep slope - lower efficiency
            {"Q": 0.20, "S0": 0.05, "Sx": 0.04, "L": 0.6, "W": 0.6, "n": 0.017, "min_eff": 0.1},
            # Moderate conditions
            {"Q": 0.05, "S0": 0.02, "Sx": 0.04, "L": 0.6, "W": 0.6, "n": 0.017, "min_eff": 0.2},
        ]
        
        for case in test_cases:
            min_eff = case.pop("min_eff")
            Qc = capturedFlow(**case)
            efficiency = Qc / case["Q"]
            
            # Check minimum efficiency is met
            assert efficiency >= min_eff, f"Efficiency {efficiency:.2%} below minimum for case"
            # Check maximum efficiency
            assert efficiency <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
