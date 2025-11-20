import pytest
import numpy as np
from app.newtonBisection import newtonBisection


class TestNewtonBisection:
    """Test suite for the newtonBisection root-finding function."""

    def test_simple_polynomial_root(self):
        """Test finding root of f(x) = x^2 - 4 (root at x=2)."""

        def func(x, p):
            f = x**2 - 4
            df = 2 * x
            return f, df

        root, n = newtonBisection(0, 3, func)
        assert np.isclose(root, 2.0, atol=1e-6)
        assert n > 0  # Should converge

    def test_linear_function(self):
        """Test finding root of f(x) = 2x - 6 (root at x=3)."""

        def func(x, p):
            f = 2 * x - 6
            df = 2
            return f, df

        root, n = newtonBisection(0, 5, func)
        assert np.isclose(root, 3.0, atol=1e-6)
        assert n > 0

    def test_cubic_function(self):
        """Test finding root of f(x) = x^3 - x - 2 (root near x=1.52)."""

        def func(x, p):
            f = x**3 - x - 2
            df = 3 * x**2 - 1
            return f, df

        root, n = newtonBisection(1, 2, func)
        # Verify it's actually a root
        f_at_root, _ = func(root, None)
        assert np.isclose(f_at_root, 0.0, atol=1e-6)
        assert n > 0

    def test_transcendental_function(self):
        """Test finding root of f(x) = cos(x) - x (root near x=0.739)."""

        def func(x, p):
            f = np.cos(x) - x
            df = -np.sin(x) - 1
            return f, df

        root, n = newtonBisection(0, 1, func)
        f_at_root, _ = func(root, None)
        assert np.isclose(f_at_root, 0.0, atol=1e-6)
        assert n > 0

    def test_with_auxiliary_parameters(self):
        """Test function that uses auxiliary parameter p."""

        def func(x, p):
            # f(x) = x^2 - p, root at x = sqrt(p)
            f = x**2 - p
            df = 2 * x
            return f, df

        p = 9.0
        root, n = newtonBisection(0, 5, func, p=p)
        assert np.isclose(root, np.sqrt(p), atol=1e-6)
        assert n > 0

    def test_custom_tolerance(self):
        """Test that custom tolerance is respected."""

        def func(x, p):
            f = x**2 - 4
            df = 2 * x
            return f, df

        # Looser tolerance
        root1, n1 = newtonBisection(0, 3, func, tol=1e-3)
        # Tighter tolerance
        root2, n2 = newtonBisection(0, 3, func, tol=1e-9)

        # Both should be close to 2, but tighter tolerance likely needs more iterations
        assert np.isclose(root1, 2.0, atol=1e-3)
        assert np.isclose(root2, 2.0, atol=1e-9)
        assert n2 >= n1  # Tighter tolerance usually needs more iterations

    def test_reversed_bracket_endpoints(self):
        """Test that algorithm handles reversed endpoints (f(x1) > f(x2))."""

        def func(x, p):
            f = x**2 - 4
            df = 2 * x
            return f, df

        # Give endpoints in "wrong" order - should be auto-corrected
        root, n = newtonBisection(3, 0, func)
        assert np.isclose(root, 2.0, atol=1e-6)
        assert n > 0

    def test_root_at_boundary(self):
        """Test when root is very close to one of the boundaries."""

        def func(x, p):
            f = (x - 1) ** 2 - 0.01  # Roots at x ≈ 0.9 and x ≈ 1.1
            df = 2 * (x - 1)
            return f, df

        root, n = newtonBisection(0.5, 0.95, func)
        f_at_root, _ = func(root, None)
        assert np.isclose(f_at_root, 0.0, atol=1e-6)

    def test_maximum_iterations_exceeded(self):
        """Test that function returns 0 when max iterations exceeded."""

        def func(x, p):
            # Very flat function that converges slowly
            f = 0.0001 * (x - 2)
            df = 0.0001
            return f, df

        root, n = newtonBisection(0, 10, func, tol=1e-12, maxit=5)
        # Should return 0 when max iterations exceeded
        assert n == 0

    def test_with_initial_guess(self):
        """Test using custom initial guess."""

        def func(x, p):
            f = x**2 - 4
            df = 2 * x
            return f, df

        # Provide good initial guess
        root, n = newtonBisection(0, 3, func, xinit=1.9)
        assert np.isclose(root, 2.0, atol=1e-6)
        assert n > 0

    def test_exponential_function(self):
        """Test finding root of f(x) = e^x - 3 (root at x=ln(3))."""

        def func(x, p):
            f = np.exp(x) - 3
            df = np.exp(x)
            return f, df

        root, n = newtonBisection(0, 2, func)
        assert np.isclose(root, np.log(3), atol=1e-6)
        assert n > 0

    def test_negative_root(self):
        """Test finding a negative root."""

        def func(x, p):
            f = x**2 - 9  # Roots at x = -3 and x = 3
            df = 2 * x
            return f, df

        root, n = newtonBisection(-5, -1, func)
        assert np.isclose(root, -3.0, atol=1e-6)
        assert n > 0

    def test_narrow_bracket(self):
        """Test with a very narrow initial bracket."""

        def func(x, p):
            f = x**2 - 4
            df = 2 * x
            return f, df

        root, n = newtonBisection(1.9, 2.1, func)
        assert np.isclose(root, 2.0, atol=1e-6)
        assert n > 0

    def test_wide_bracket(self):
        """Test with a very wide initial bracket."""

        def func(x, p):
            f = x**2 - 4
            df = 2 * x
            return f, df

        root, n = newtonBisection(-10, 10, func)
        # Should find the positive root (x=2) since it's the first one encountered
        f_at_root, _ = func(root, None)
        assert np.isclose(f_at_root, 0.0, atol=1e-6)
        assert n > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
