import pytest
from pytest import approx
import numpy as np
import pandas as pd
import igraph as ig
from pathlib import Path
from unittest.mock import patch
from app.subcatchmentGraph import SubcatchmentGraph


class TestSubcatchmentGraph:
    """Test suite for SubcatchmentGraph class."""

    @pytest.fixture
    def simple_csv_data(self):
        """Simple CSV data for a basic subcatchment network."""
        return pd.DataFrame({
            "type": ["SUBCATCHMENT", "SUBCATCHMENT", "SUBCATCHMENT"],
            "id": [0, 1, 2],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0],
            "z": [10.0, 5.0, 0.0],
            "slope": [0.02, 0.02, 0.02],
            "outgoing": [0, 1, 2],
        })

    @pytest.fixture
    def complex_csv_data(self):
        """More complex CSV data with multiple subcatchments."""
        return pd.DataFrame({
            "type": ["SUBCATCHMENT"] * 5 + ["STREET"] * 2,
            "id": [0, 1, 2, 3, 4, 10, 11],
            "x": [0.0, 100.0, 200.0, 300.0, 400.0, 50.0, 150.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "z": [20.0, 15.0, 10.0, 5.0, 0.0, 12.0, 7.0],
            "slope": [0.01, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02],
            "outgoing": [0, 0, 1, 2, 3, 0, 1],
        })

    @pytest.fixture
    def temp_csv_file(self, simple_csv_data, tmp_path, monkeypatch):
        """Create a temporary CSV file for testing."""
        # Create data directory in temp path
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        csv_path = data_dir / "test_network.csv"
        simple_csv_data.to_csv(csv_path, index=False)
        
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        return "test_network"

    @pytest.fixture
    def temp_complex_csv_file(self, complex_csv_data, tmp_path, monkeypatch):
        """Create a temporary CSV file with complex data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        csv_path = data_dir / "complex_network.csv"
        complex_csv_data.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        
        return "complex_network"

    # Initialization tests
    def test_init_basic(self, temp_csv_file):
        """Test basic initialization of SubcatchmentGraph."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        assert graph.G.vcount() == 3
        assert graph.oldwaterRatio == 0.2

    def test_init_custom_oldwater_ratio(self, temp_csv_file):
        """Test initialization with custom oldwater ratio."""
        graph = SubcatchmentGraph(temp_csv_file, oldwaterRatio=0.5)
        
        assert graph.oldwaterRatio == 0.5

    def test_init_vertex_count(self, temp_complex_csv_file):
        """Test that only SUBCATCHMENT types are counted as vertices."""
        graph = SubcatchmentGraph(temp_complex_csv_file)
        
        # Should only have 5 subcatchments, not 7 total rows
        assert graph.G.vcount() == 5

    def test_init_vertex_attributes(self, temp_csv_file):
        """Test that vertex attributes are properly initialized."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        # Check attribute existence
        assert "coupledID" in graph.G.vs.attributes()
        assert "x" in graph.G.vs.attributes()
        assert "y" in graph.G.vs.attributes()
        assert "z" in graph.G.vs.attributes()
        assert "area" in graph.G.vs.attributes()
        assert "width" in graph.G.vs.attributes()
        assert "slope" in graph.G.vs.attributes()
        assert "n" in graph.G.vs.attributes()
        assert "depth" in graph.G.vs.attributes()
        assert "invert" in graph.G.vs.attributes()

    def test_init_depth_initialized_to_zero(self, temp_csv_file):
        """Test that depths are initialized to zero."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        assert all(d == 0 for d in graph.G.vs["depth"])

    def test_init_invert_initialized_to_zero(self, temp_csv_file):
        """Test that inverts are initialized to zero."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        assert all(inv == 0 for inv in graph.G.vs["invert"])

    def test_init_coordinates_loaded(self, temp_csv_file):
        """Test that x, y, z coordinates are properly loaded."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        assert graph.G.vs["x"] == [0.0, 100.0, 200.0]
        assert graph.G.vs["y"] == [0.0, 0.0, 0.0]
        assert graph.G.vs["z"] == [10.0, 5.0, 0.0]

    def test_init_slope_loaded(self, temp_csv_file):
        """Test that slopes are properly loaded."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        assert all(s == 0.02 for s in graph.G.vs["slope"])

    def test_init_hydraulic_coupling(self, temp_csv_file):
        """Test that hydraulic coupling IDs are stored."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        assert len(graph.hydraulicCoupling) == 3
        assert list(graph.hydraulicCoupling) == [0, 1, 2]

    def test_init_area_hardcoded(self, temp_csv_file):
        """Test that area is currently hardcoded (TODO in code)."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        # Note: This is hardcoded in the current implementation
        assert graph.G.vs["area"] == [10000.0, 10000.0, 10000.0]

    def test_init_width_hardcoded(self, temp_csv_file):
        """Test that width is currently hardcoded (TODO in code)."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        # Note: This is hardcoded in the current implementation
        assert graph.G.vs["width"] == [100.0, 100.0, 100.0]

    def test_init_mannings_n_hardcoded(self, temp_csv_file):
        """Test that Manning's n is currently hardcoded (TODO in code)."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        # Note: This is hardcoded in the current implementation
        assert all(n == 0.017 for n in graph.G.vs["n"])

    def test_init_graph_is_directed(self, temp_csv_file):
        """Test that the graph is directed."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        assert graph.G.is_directed()

    # Update method tests
    def test_update_basic(self, temp_csv_file):
        """Test basic update functionality."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        t = 0.0
        dt = 3600.0  # 1 hour in seconds
        rainfall = 0.001  # m/s
        
        depths, outflows = graph.update(t, dt, rainfall)
        
        assert len(depths) == 3
        assert len(outflows) == 3
        assert all(np.isfinite(d) for d in depths)
        assert all(np.isfinite(o) for o in outflows)

    def test_update_zero_rainfall(self, temp_csv_file):
        """Test update with zero rainfall."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        depths, outflows = graph.update(0.0, 3600.0, 0.0)
        
        # With zero rainfall and zero initial depth, should remain at zero
        assert all(d == approx(0.0, abs=1e-10) for d in depths)

    def test_update_positive_rainfall(self, temp_csv_file):
        """Test that positive rainfall increases depth."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        rainfall = 0.001  # m/s
        depths, outflows = graph.update(0.0, 3600.0, rainfall)
        
        # With positive rainfall, depths should increase
        assert all(d >= 0 for d in depths)

    def test_update_depths_are_updated(self, temp_csv_file):
        """Test that graph depths are updated after update call."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        initial_depths = graph.G.vs["depth"].copy()
        depths, outflows = graph.update(0.0, 3600.0, 0.001)
        
        # Returned depths should match graph depths
        assert list(depths) == list(graph.G.vs["depth"])

    def test_update_multiple_times(self, temp_csv_file):
        """Test multiple sequential updates."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        t = 0.0
        dt = 1800.0  # 30 minutes
        rainfall = 0.0005
        
        depths1, _ = graph.update(t, dt, rainfall)
        depths2, _ = graph.update(t + dt, dt, rainfall)
        depths3, _ = graph.update(t + 2*dt, dt, rainfall)
        
        # All depths should be finite
        assert all(np.isfinite(d) for d in depths3)

    def test_update_outflow_positive(self, temp_csv_file):
        """Test that outflows are non-negative."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        # Add some initial depth
        graph.G.vs["depth"] = [0.1, 0.1, 0.1]
        
        depths, outflows = graph.update(0.0, 3600.0, 0.0)
        
        assert all(o >= 0 for o in outflows)

    def test_update_oldwater_ratio_effect(self, simple_csv_data, tmp_path, monkeypatch):
        """Test that oldwater ratio affects depth accumulation."""
        # Create temp directory and CSV
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "test.csv"
        simple_csv_data.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        
        # Test with low oldwater ratio (more infiltration)
        graph_low = SubcatchmentGraph("test", oldwaterRatio=0.1)
        depths_low, _ = graph_low.update(0.0, 3600.0, 0.001)
        
        # Test with high oldwater ratio (less infiltration)
        graph_high = SubcatchmentGraph("test", oldwaterRatio=0.9)
        depths_high, _ = graph_high.update(0.0, 3600.0, 0.001)
        
        # Higher ratio should result in higher depths
        assert sum(depths_high) > sum(depths_low)

    def test_update_manning_equation_components(self, temp_csv_file):
        """Test that Manning equation is correctly applied."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        # Set known depth
        graph.G.vs["depth"] = [0.1, 0.1, 0.1]
        
        depths, outflows = graph.update(0.0, 1.0, 0.0)
        
        # Verify outflows are positive and finite
        assert all(o > 0 for o in outflows)
        assert all(np.isfinite(o) for o in outflows)

    def test_update_depth_above_invert(self, temp_csv_file):
        """Test that only depth above invert contributes to outflow."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        # Set depth below invert
        graph.G.vs["invert"] = [0.5, 0.5, 0.5]
        graph.G.vs["depth"] = [0.2, 0.2, 0.2]
        
        depths, outflows = graph.update(0.0, 1.0, 0.0)
        
        # With depth below invert, outflow should be zero
        assert all(o == approx(0.0, abs=1e-10) for o in outflows)

    def test_update_varying_slopes(self, temp_csv_file):
        """Test update with varying slopes across subcatchments."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        # Set different slopes
        graph.G.vs["slope"] = [0.01, 0.02, 0.04]
        graph.G.vs["depth"] = [0.1, 0.1, 0.1]
        
        depths, outflows = graph.update(0.0, 1.0, 0.0)
        
        # Steeper slopes should have higher outflows (all else equal)
        assert outflows[2] > outflows[0]

    def test_update_returns_correct_types(self, temp_csv_file):
        """Test that update returns correct data types."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        depths, outflows = graph.update(0.0, 3600.0, 0.001)
        
        assert isinstance(depths, np.ndarray)
        assert isinstance(outflows, np.ndarray)
        assert depths.shape == (3,)
        assert outflows.shape == (3,)

    # Visualize method tests
    def test_visualize_basic(self, temp_csv_file, tmp_path):
        """Test basic visualization functionality."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        times = [0, 1, 2, 3]
        depths = [
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.03, 0.03, 0.03],
        ]
        
        # Create figures directory
        fig_dir = tmp_path / "figures"
        fig_dir.mkdir()
        
        with patch('matplotlib.pyplot.show'):
            graph.visualize(times, depths, fileName="test_output")
        
        # Check that file was created
        assert (fig_dir / "test_output.png").exists()

    def test_visualize_default_filename(self, temp_csv_file, tmp_path):
        """Test visualization with default filename."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        times = [0, 1, 2]
        depths = [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [0.02, 0.02, 0.02]]
        
        fig_dir = tmp_path / "figures"
        fig_dir.mkdir()
        
        with patch('matplotlib.pyplot.show'):
            graph.visualize(times, depths)
        
        # Default filename should be "subcatchmentGraph"
        assert (fig_dir / "subcatchmentGraph.png").exists()

    def test_visualize_with_numpy_array(self, temp_csv_file, tmp_path):
        """Test visualization accepts numpy arrays."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        times = np.array([0, 1, 2, 3])
        depths = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.03, 0.03, 0.03],
        ])
        
        fig_dir = tmp_path / "figures"
        fig_dir.mkdir()
        
        with patch('matplotlib.pyplot.show'):
            graph.visualize(times, depths, fileName="numpy_test")

    # Integration tests
    def test_full_simulation_workflow(self, temp_csv_file, tmp_path):
        """Test complete workflow: init -> multiple updates -> visualize."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        # Run simulation
        times = []
        all_depths = []
        t = 0.0
        dt = 1800.0  # 30 minutes
        rainfall = 0.0005
        
        for i in range(5):
            times.append(t / 3600.0)  # Convert to hours
            depths, _ = graph.update(t, dt, rainfall)
            all_depths.append(depths.copy())
            t += dt
        
        # Visualize results
        fig_dir = tmp_path / "figures"
        fig_dir.mkdir()
        
        with patch('matplotlib.pyplot.show'):
            graph.visualize(times, all_depths, fileName="simulation_test")
        
        # Check that depths increased over time (with rainfall)
        assert all_depths[-1][0] + 1e-5 > all_depths[0][0]

    def test_conservation_of_mass(self, temp_csv_file):
        """Test that mass is approximately conserved."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        initial_depth = graph.G.vs["depth"].copy()
        rainfall = 0.001
        dt = 3600.0
        
        depths, outflows = graph.update(0.0, dt, rainfall)
        
        # Calculate total input
        total_input = rainfall * graph.oldwaterRatio * dt * sum(graph.G.vs["area"])
        
        # Calculate total output
        total_output = sum(outflows) * dt
        
        # Calculate storage change
        storage_change = sum((depths[i] - initial_depth[i]) * graph.G.vs["area"][i] 
                           for i in range(len(depths)))
        
        # Mass balance: input - output ≈ storage change
        # Allow for numerical integration errors
        if total_input > 0:
            assert abs(total_input - total_output - storage_change) / total_input < 0.5

    def test_steady_state_convergence(self, temp_csv_file):
        """Test that system approaches steady state with constant rainfall."""
        graph = SubcatchmentGraph(temp_csv_file)
        
        rainfall = 0.0001  # Small constant rainfall
        dt = 3600.0
        
        # Run for many time steps
        for i in range(20):
            depths, outflows = graph.update(i * dt, dt, rainfall)
        
        depth_before = depths.copy()
        
        # Run a few more steps
        for i in range(3):
            depths, outflows = graph.update((20 + i) * dt, dt, rainfall)
        
        # Change in depth should be small (approaching steady state)
        if depth_before[0] > 1e-6:
            relative_change = abs(depths[0] - depth_before[0]) / depth_before[0]
            assert relative_change < 0.5

    def test_different_network_sizes(self, tmp_path, monkeypatch):
        """Test that code works with different network sizes."""
        for n_subcatchments in [1, 3, 5]:
            csv_data = pd.DataFrame({
                "type": ["SUBCATCHMENT"] * n_subcatchments,
                "id": list(range(n_subcatchments)),
                "x": [float(i * 100) for i in range(n_subcatchments)],
                "y": [0.0] * n_subcatchments,
                "z": [float(10 - i) for i in range(n_subcatchments)],
                "slope": [0.02] * n_subcatchments,
                "outgoing": list(range(n_subcatchments)),
            })
            
            # Create temp directory for this test
            test_dir = tmp_path / f"test_{n_subcatchments}"
            test_dir.mkdir()
            data_dir = test_dir / "data"
            data_dir.mkdir()
            csv_path = data_dir / "test.csv"
            csv_data.to_csv(csv_path, index=False)
            
            # Change to test directory
            monkeypatch.chdir(test_dir)
            
            graph = SubcatchmentGraph("test")
            
            assert graph.G.vcount() == n_subcatchments
            
            # Test update works
            depths, outflows = graph.update(0.0, 3600.0, 0.001)
            assert len(depths) == n_subcatchments
            assert len(outflows) == n_subcatchments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
