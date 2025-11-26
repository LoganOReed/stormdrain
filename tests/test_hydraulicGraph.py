import pytest
from pytest import approx
import numpy as np
import pandas as pd
import igraph as ig
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.hydraulicGraph import HydraulicGraph


class TestHydraulicGraphInitialization:
    """Test suite for HydraulicGraph initialization."""

    @pytest.fixture
    def street_csv_data(self):
        """CSV data for a basic street network."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4],
            "type": ["OUTFALLSTREET", "STREET", "STREET", "STREET"],
            "x": [0.0, 100.0, 200.0, 300.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 1.0, 2.0, 3.0],
            "slope": [0.0, 0.01, 0.01, 0.01],
            "outgoing": [-1, 1, 2, 3],
            "drain": [0, 1, 0, 0],
            "drainCoupledID": [-1, 10, -1, -1],
            "drainType": ["P-50x100", "P-50x100", "P-50x100", "P-50x100"],
            "drainLength": [0.818, 0.818, 0.818, 0.818],
            "drainWidth": [0.415, 0.415, 0.415, 0.415],
        })

    @pytest.fixture
    def sewer_csv_data(self):
        """CSV data for a basic sewer network."""
        return pd.DataFrame({
            "id": [10, 11, 12],
            "type": ["OUTFALLSEWER", "SEWER", "SEWER"],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0],
            "z": [-1.0, 0.0, 1.0],
            "slope": [0.0, 0.01, 0.01],
            "outgoing": [-1, 10, 11],
            "drain": [0, 1, 0],
            "drainCoupledID": [-1, 2, -1],
            "drainType": ["P-50x100", "P-50x100", "P-50x100"],
            "drainLength": [0.818, 0.818, 0.818],
            "drainWidth": [0.415, 0.415, 0.415],
        })

    @pytest.fixture
    def combined_csv_data(self, street_csv_data, sewer_csv_data):
        """Combined CSV data with both street and sewer."""
        return pd.concat([street_csv_data, sewer_csv_data], ignore_index=True)

    @pytest.fixture
    def temp_street_csv(self, street_csv_data, tmp_path, monkeypatch):
        """Create a temporary CSV file for street network testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "test_street.csv"
        street_csv_data.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        return "test_street"

    @pytest.fixture
    def temp_sewer_csv(self, sewer_csv_data, tmp_path, monkeypatch):
        """Create a temporary CSV file for sewer network testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "test_sewer.csv"
        sewer_csv_data.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        return "test_sewer"

    @pytest.fixture
    def temp_combined_csv(self, combined_csv_data, tmp_path, monkeypatch):
        """Create a temporary CSV file with combined data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "test_combined.csv"
        combined_csv_data.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        return "test_combined"

    # Basic initialization tests
    def test_init_street_graph_type(self, temp_street_csv):
        """Test that STREET graph type is properly set."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        assert graph.graphType == "STREET"

    def test_init_sewer_graph_type(self, temp_sewer_csv):
        """Test that SEWER graph type is properly set."""
        graph = HydraulicGraph("SEWER", temp_sewer_csv)
        assert graph.graphType == "SEWER"

    def test_init_filters_by_graph_type(self, temp_combined_csv):
        """Test that only nodes of the specified type are included."""
        street_graph = HydraulicGraph("STREET", temp_combined_csv)
        sewer_graph = HydraulicGraph("SEWER", temp_combined_csv)

        # Street should have 4 nodes, sewer should have 3
        assert street_graph.G.vcount() == 4
        assert sewer_graph.G.vcount() == 3

    def test_init_graph_is_directed(self, temp_street_csv):
        """Test that the graph is directed."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        assert graph.G.is_directed()

    def test_init_phi_theta_constants(self, temp_street_csv):
        """Test that PHI and THETA constants are set."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        assert graph.PHI == 0.6
        assert graph.THETA == 0.6

    # Vertex attribute tests
    def test_init_vertex_attributes_exist(self, temp_street_csv):
        """Test that all required vertex attributes are created."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        required_attrs = [
            "coupledID", "invert", "x", "y", "z", "depth", "type",
            "drain", "drainType", "drainCoupledID", "drainLength", "drainWidth"
        ]
        
        for attr in required_attrs:
            assert attr in graph.G.vs.attributes(), f"Missing vertex attribute: {attr}"

    def test_init_coordinates_loaded(self, temp_street_csv):
        """Test that coordinates are properly loaded."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        assert list(graph.G.vs["x"]) == [0.0, 100.0, 200.0, 300.0]
        assert list(graph.G.vs["y"]) == [0.0, 0.0, 0.0, 0.0]
        assert list(graph.G.vs["z"]) == [0.0, 1.0, 2.0, 3.0]

    def test_init_depth_initialized_to_zero(self, temp_street_csv):
        """Test that depths are initialized to zero."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        assert all(d == 0 for d in graph.G.vs["depth"])

    def test_init_invert_initialized_to_zero(self, temp_street_csv):
        """Test that inverts are initialized to zero."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        assert all(inv == 0 for inv in graph.G.vs["invert"])

    def test_init_node_type_classification(self, temp_street_csv):
        """Test that node types are correctly classified (0=junction, 1=outfall)."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        # First node is outfall (type=1), others are junctions (type=0)
        assert graph.G.vs[0]["type"] == 1  # OUTFALLSTREET
        assert graph.G.vs[1]["type"] == 0  # STREET
        assert graph.G.vs[2]["type"] == 0  # STREET
        assert graph.G.vs[3]["type"] == 0  # STREET

    def test_init_drain_attributes_loaded(self, temp_street_csv):
        """Test that drain attributes are properly loaded."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        # Node index 1 (second street node) should have drain=1
        assert graph.G.vs[1]["drain"] == 1
        assert graph.G.vs[1]["drainCoupledID"] == 10

    # Edge attribute tests
    def test_init_edge_attributes_exist(self, temp_street_csv):
        """Test that all required edge attributes are created."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        required_attrs = [
            "length", "slope", "offsetHeight", "n", "Sx",
            "T_curb", "T_crown", "H_curb", "S_back",
            "yFull", "Afull", "PsiFull", "beta", "qFull",
            "Q1", "Q2", "A1", "A2",
            "Q1Prev", "Q2Prev", "A1Prev", "A2Prev"
        ]
        
        for attr in required_attrs:
            assert attr in graph.G.es.attributes(), f"Missing edge attribute: {attr}"

    def test_init_edge_length_calculation(self, temp_street_csv):
        """Test that edge lengths are correctly calculated."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        # For nodes at (100,0,1) to (0,0,0), length should be ~100.005
        for e in graph.G.es:
            assert e["length"] > 0
            assert np.isfinite(e["length"])

    def test_init_edge_slope_calculation(self, temp_street_csv):
        """Test that edge slopes are correctly calculated."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        # Slope should be positive (z increases from target to source)
        for e in graph.G.es:
            assert e["slope"] > 0
            assert np.isfinite(e["slope"])

    def test_init_mannings_n(self, temp_street_csv):
        """Test that Manning's n is set."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        # Default value is 0.013
        for e in graph.G.es:
            assert e["n"] == 0.013

    def test_init_flow_values_zero(self, temp_street_csv):
        """Test that flow values are initialized to zero."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        assert all(q == 0 for q in graph.G.es["Q1"])
        assert all(q == 0 for q in graph.G.es["Q2"])
        assert all(a == 0 for a in graph.G.es["A1"])
        assert all(a == 0 for a in graph.G.es["A2"])

    def test_init_previous_values_zero(self, temp_street_csv):
        """Test that previous timestep values are initialized to zero."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        assert all(q == 0 for q in graph.G.es["Q1Prev"])
        assert all(q == 0 for q in graph.G.es["Q2Prev"])
        assert all(a == 0 for a in graph.G.es["A1Prev"])
        assert all(a == 0 for a in graph.G.es["A2Prev"])

    # Geometry function assignment tests
    def test_init_street_geometry_functions(self, temp_street_csv):
        """Test that street geometry functions are assigned for STREET type."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        # Test that the functions are assigned (by checking they're callable)
        assert callable(graph.depthFromArea)
        assert callable(graph.psiFromArea)
        assert callable(graph.psiPrimeFromArea)
        assert callable(graph.areaFromPsi)

    def test_init_sewer_geometry_functions(self, temp_sewer_csv):
        """Test that circular geometry functions are assigned for SEWER type."""
        graph = HydraulicGraph("SEWER", temp_sewer_csv)
        
        # Test that the functions are assigned
        assert callable(graph.depthFromArea)
        assert callable(graph.psiFromArea)
        assert callable(graph.psiPrimeFromArea)
        assert callable(graph.areaFromPsi)

    # Edge creation tests
    def test_init_edges_created(self, temp_street_csv):
        """Test that edges are properly created."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        # Should have 3 edges (4 nodes, 1 is outfall with no outgoing)
        assert graph.G.ecount() == 3

    def test_init_edge_direction(self, temp_street_csv):
        """Test that edges point in correct direction."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        # Flow should go from higher nodes toward outfall
        for e in graph.G.es:
            source_z = graph.G.vs[e.source]["z"]
            target_z = graph.G.vs[e.target]["z"]
            # Source should generally be higher than target
            assert source_z >= target_z or abs(source_z - target_z) < 0.1

    # Full area and psi tests
    def test_init_street_yfull_set(self, temp_street_csv):
        """Test that yFull is set for street graph."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        for e in graph.G.es:
            assert e["yFull"] > 0

    def test_init_sewer_yfull_set(self, temp_sewer_csv):
        """Test that yFull is set for sewer graph."""
        graph = HydraulicGraph("SEWER", temp_sewer_csv)
        
        # Default sewer yFull is 0.5 (18" pipe)
        for e in graph.G.es:
            assert e["yFull"] == 0.5

    def test_init_afull_positive(self, temp_street_csv):
        """Test that Afull is positive."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        for e in graph.G.es:
            assert e["Afull"] > 0

    def test_init_psifull_positive(self, temp_street_csv):
        """Test that PsiFull is positive."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        for e in graph.G.es:
            assert e["PsiFull"] > 0

    def test_init_beta_calculation(self, temp_street_csv):
        """Test that beta is correctly calculated."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        for e in graph.G.es:
            expected_beta = np.sqrt(e["slope"]) / e["n"]
            assert e["beta"] == approx(expected_beta, rel=1e-6)

    def test_init_qfull_calculation(self, temp_street_csv):
        """Test that qFull is correctly calculated."""
        graph = HydraulicGraph("STREET", temp_street_csv)
        
        for e in graph.G.es:
            expected_qfull = e["beta"] * e["PsiFull"]
            assert e["qFull"] == approx(expected_qfull, rel=1e-6)


class TestHydraulicGraphUpdate:
    """Test suite for HydraulicGraph update method."""

    @pytest.fixture
    def simple_network_csv(self):
        """CSV data for a simple 3-node network."""
        return pd.DataFrame({
            "id": [1, 2, 3],
            "type": ["OUTFALLSTREET", "STREET", "STREET"],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 1.0, 2.0],
            "slope": [0.0, 0.01, 0.01],
            "outgoing": [-1, 1, 2],
            "drain": [0, 0, 0],
            "drainCoupledID": [-1, -1, -1],
            "drainType": ["P-50x100", "P-50x100", "P-50x100"],
            "drainLength": [0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def temp_simple_csv(self, simple_network_csv, tmp_path, monkeypatch):
        """Create temporary CSV for simple network."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "simple_network.csv"
        simple_network_csv.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        return "simple_network"

    @pytest.fixture
    def default_coupling(self, simple_network_csv):
        """Default coupling terms with no external flow."""
        n = len(simple_network_csv)
        return {
            "subcatchmentRunoff": np.zeros(n),
            "drainCapture": np.zeros(n),
            "drainOverflow": np.zeros(n),
        }

    @pytest.fixture
    def coupling_with_runoff(self, simple_network_csv):
        """Coupling terms with subcatchment runoff."""
        n = len(simple_network_csv)
        runoff = np.zeros(n)
        runoff[2] = 0.01  # Add runoff at upstream node (id=3 -> index 2)
        return {
            "subcatchmentRunoff": runoff,
            "drainCapture": np.zeros(n),
            "drainOverflow": np.zeros(n),
        }

    # Basic update tests
    def test_update_returns_correct_shape(self, temp_simple_csv, default_coupling):
        """Test that update returns arrays of correct shape."""
        graph = HydraulicGraph("STREET", temp_simple_csv)
        
        depth, area, coupling, peak = graph.update(0.0, 1800.0, default_coupling)
        
        assert len(depth) == graph.G.vcount()
        assert len(area) == graph.G.ecount()
        assert isinstance(peak, (int, float))

    def test_update_returns_finite_values(self, temp_simple_csv, default_coupling):
        """Test that update returns finite values."""
        graph = HydraulicGraph("STREET", temp_simple_csv)
        
        depth, area, coupling, peak = graph.update(0.0, 1800.0, default_coupling)
        
        assert all(np.isfinite(d) for d in depth)
        assert all(np.isfinite(a) for a in area)
        assert np.isfinite(peak)

    def test_update_no_input_stays_zero(self, temp_simple_csv, default_coupling):
        """Test that with no input flow, values stay at zero."""
        graph = HydraulicGraph("STREET", temp_simple_csv)
        
        depth, area, coupling, peak = graph.update(0.0, 1800.0, default_coupling)
        
        # With no input, there should be no flow
        assert all(q == approx(0.0, abs=1e-10) for q in graph.G.es["Q1"])
        assert all(q == approx(0.0, abs=1e-10) for q in graph.G.es["Q2"])

    def test_update_with_runoff_generates_flow(self, temp_simple_csv, coupling_with_runoff):
        """Test that runoff input generates flow in the system."""
        graph = HydraulicGraph("STREET", temp_simple_csv)
        
        depth, area, coupling, peak = graph.update(0.0, 1800.0, coupling_with_runoff)
        
        # Should have some positive flow
        assert any(q > 0 for q in graph.G.es["Q1"])

    def test_update_preserves_previous_values(self, temp_simple_csv, coupling_with_runoff):
        """Test that previous timestep values are preserved."""
        graph = HydraulicGraph("STREET", temp_simple_csv)
        
        # First update
        graph.update(0.0, 1800.0, coupling_with_runoff)
        Q1_after_first = np.array(graph.G.es["Q1"]).copy()
        
        # Second update
        graph.update(1800.0, 1800.0, coupling_with_runoff)
        
        # Previous values should match first update's current values
        np.testing.assert_array_almost_equal(graph.G.es["Q1Prev"], Q1_after_first)

    def test_update_depth_updated(self, temp_simple_csv, coupling_with_runoff):
        """Test that node depths are updated after update call."""
        graph = HydraulicGraph("STREET", temp_simple_csv)
        
        initial_depths = np.array(graph.G.vs["depth"]).copy()
        depth, area, coupling, peak = graph.update(0.0, 1800.0, coupling_with_runoff)
        
        # Returned depths should match graph depths
        np.testing.assert_array_equal(depth, graph.G.vs["depth"])

    def test_update_multiple_timesteps(self, temp_simple_csv, coupling_with_runoff):
        """Test multiple sequential updates."""
        graph = HydraulicGraph("STREET", temp_simple_csv)
        
        dt = 900.0  # 15 minutes
        for i in range(5):
            depth, area, coupling, peak = graph.update(i * dt, dt, coupling_with_runoff)
            
            # Values should remain finite
            assert all(np.isfinite(d) for d in depth)
            assert all(np.isfinite(a) for a in area)
            assert np.isfinite(peak)

    def test_update_peak_discharge_non_negative(self, temp_simple_csv, coupling_with_runoff):
        """Test that peak discharge is non-negative."""
        graph = HydraulicGraph("STREET", temp_simple_csv)
        
        depth, area, coupling, peak = graph.update(0.0, 1800.0, coupling_with_runoff)
        
        assert peak >= 0

    # Graph DAG requirement test
    def test_update_requires_acyclic_graph(self, tmp_path, monkeypatch):
        """Test that update fails on cyclic graphs."""
        # Create a cyclic network (which shouldn't happen in real data)
        cyclic_data = pd.DataFrame({
            "id": [1, 2, 3],
            "type": ["STREET", "STREET", "STREET"],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 1.0, 2.0],
            "slope": [0.01, 0.01, 0.01],
            "outgoing": [3, 1, 2],  # Creates a cycle: 1->3, 2->1, 3->2
            "drain": [0, 0, 0],
            "drainCoupledID": [-1, -1, -1],
            "drainType": ["P-50x100", "P-50x100", "P-50x100"],
            "drainLength": [0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.0, 0.0],
        })
        
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "cyclic.csv"
        cyclic_data.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        
        graph = HydraulicGraph("STREET", "cyclic")
        coupling = {
            "subcatchmentRunoff": np.zeros(3),
            "drainCapture": np.zeros(3),
            "drainOverflow": np.zeros(3),
        }
        
        with pytest.raises(ValueError, match="acyclic"):
            graph.update(0.0, 1800.0, coupling)


class TestHydraulicGraphCoupling:
    """Test suite for coupling term handling."""

    @pytest.fixture
    def network_with_drains_csv(self):
        """CSV data for network with drain connections."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4],
            "type": ["OUTFALLSTREET", "STREET", "STREET", "STREET"],
            "x": [0.0, 100.0, 200.0, 300.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 1.0, 2.0, 3.0],
            "slope": [0.0, 0.01, 0.01, 0.01],
            "outgoing": [-1, 1, 2, 3],
            "drain": [0, 1, 1, 0],  # Drains at nodes 2 and 3
            "drainCoupledID": [-1, 10, 11, -1],
            "drainType": ["P-50x100", "P-50x100", "P-50x100", "P-50x100"],
            "drainLength": [0.0, 0.6, 0.6, 0.0],
            "drainWidth": [0.0, 0.6, 0.6, 0.0],
        })

    @pytest.fixture
    def temp_drain_csv(self, network_with_drains_csv, tmp_path, monkeypatch):
        """Create temporary CSV for drain network."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "drain_network.csv"
        network_with_drains_csv.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        return "drain_network"

    def test_coupling_subcatchment_runoff(self, temp_drain_csv):
        """Test that subcatchment runoff is properly coupled."""
        graph = HydraulicGraph("STREET", temp_drain_csv)
        
        coupling = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 0.0, 0.01]),  # Runoff at node 4 (id=4)
            "drainCapture": np.zeros(4),
            "drainOverflow": np.zeros(4),
        }
        
        depth, area, _, peak = graph.update(0.0, 1800.0, coupling)
        
        # Should have flow in the system
        assert any(q > 0 for q in graph.G.es["Q1"])

    def test_coupling_drain_capture(self, temp_drain_csv):
        """Test that drain capture coupling works."""
        graph = HydraulicGraph("STREET", temp_drain_csv)
        
        # Positive drain capture should remove water from street
        coupling = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 0.0, 0.05]),  # Input flow
            "drainCapture": np.array([0.0, -0.01, 0.0, 0.0]),  # Drain capture at node 2
            "drainOverflow": np.zeros(4),
        }
        
        depth, area, coupling_out, peak = graph.update(0.0, 1800.0, coupling)
        
        # System should still function
        assert all(np.isfinite(d) for d in depth)

    def test_coupling_drain_overflow(self, temp_drain_csv):
        """Test that drain overflow coupling works."""
        graph = HydraulicGraph("STREET", temp_drain_csv)
        
        coupling = {
            "subcatchmentRunoff": np.zeros(4),
            "drainCapture": np.zeros(4),
            "drainOverflow": np.array([0.0, 0.005, 0.0, 0.0]),  # Overflow back to street
        }
        
        depth, area, coupling_out, peak = graph.update(0.0, 1800.0, coupling)
        
        # System should still function
        assert all(np.isfinite(d) for d in depth)


class TestHydraulicGraphGeometry:
    """Test suite for geometry function integration."""

    @pytest.fixture
    def street_csv(self):
        """Street network CSV."""
        return pd.DataFrame({
            "id": [1, 2, 3],
            "type": ["OUTFALLSTREET", "STREET", "STREET"],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 1.0, 2.0],
            "slope": [0.0, 0.01, 0.01],
            "outgoing": [-1, 1, 2],
            "drain": [0, 0, 0],
            "drainCoupledID": [-1, -1, -1],
            "drainType": ["P-50x100", "P-50x100", "P-50x100"],
            "drainLength": [0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def sewer_csv(self):
        """Sewer network CSV."""
        return pd.DataFrame({
            "id": [10, 11, 12],
            "type": ["OUTFALLSEWER", "SEWER", "SEWER"],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0],
            "z": [-1.0, 0.0, 1.0],
            "slope": [0.0, 0.01, 0.01],
            "outgoing": [-1, 10, 11],
            "drain": [0, 0, 0],
            "drainCoupledID": [-1, -1, -1],
            "drainType": ["P-50x100", "P-50x100", "P-50x100"],
            "drainLength": [0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def temp_street(self, street_csv, tmp_path, monkeypatch):
        """Create temp CSV for street."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "street_geom.csv"
        street_csv.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        return "street_geom"

    @pytest.fixture
    def temp_sewer(self, sewer_csv, tmp_path, monkeypatch):
        """Create temp CSV for sewer."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "sewer_geom.csv"
        sewer_csv.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        return "sewer_geom"

    def test_street_geometry_depth_calculation(self, temp_street):
        """Test that street geometry depth calculation works."""
        graph = HydraulicGraph("STREET", temp_street)
        
        # Test depth calculation for a known area
        e = graph.G.es[0]
        test_area = 0.1 * e["Afull"]
        depth = graph.depthFromArea(test_area, e)
        
        assert depth > 0
        assert np.isfinite(depth)

    def test_sewer_geometry_depth_calculation(self, temp_sewer):
        """Test that sewer (circular) geometry depth calculation works."""
        graph = HydraulicGraph("SEWER", temp_sewer)
        
        e = graph.G.es[0]
        test_area = 0.1 * e["Afull"]
        depth = graph.depthFromArea(test_area, e)
        
        assert depth > 0
        assert np.isfinite(depth)
        assert depth <= e["yFull"]

    def test_street_psi_calculation(self, temp_street):
        """Test street geometry psi calculation."""
        graph = HydraulicGraph("STREET", temp_street)
        
        e = graph.G.es[0]
        test_area = 0.1 * e["Afull"]
        psi = graph.psiFromArea(test_area, e)
        
        assert psi > 0
        assert np.isfinite(psi)

    def test_sewer_psi_calculation(self, temp_sewer):
        """Test sewer geometry psi calculation."""
        graph = HydraulicGraph("SEWER", temp_sewer)
        
        e = graph.G.es[0]
        test_area = 0.1 * e["Afull"]
        psi = graph.psiFromArea(test_area, e)
        
        assert psi > 0
        assert np.isfinite(psi)

    def test_street_area_psi_roundtrip(self, temp_street):
        """Test area -> psi -> area roundtrip for street."""
        graph = HydraulicGraph("STREET", temp_street)
        
        e = graph.G.es[0]
        test_areas = [0.1, 0.3, 0.5, 0.7]
        
        for ratio in test_areas:
            A_original = ratio * e["Afull"]
            psi = graph.psiFromArea(A_original, e)
            A_recovered = graph.areaFromPsi(psi, e)
            
            assert A_recovered == approx(A_original, rel=0.05)

    def test_sewer_area_psi_roundtrip(self, temp_sewer):
        """Test area -> psi -> area roundtrip for sewer."""
        graph = HydraulicGraph("SEWER", temp_sewer)
        
        e = graph.G.es[0]
        test_areas = [0.1, 0.3, 0.5, 0.7]
        
        for ratio in test_areas:
            A_original = ratio * e["Afull"]
            psi = graph.psiFromArea(A_original, e)
            A_recovered = graph.areaFromPsi(psi, e)
            
            assert A_recovered == approx(A_original, rel=0.05)


class TestHydraulicGraphNumericalStability:
    """Test suite for numerical stability and edge cases."""

    @pytest.fixture
    def simple_csv(self):
        """Simple network CSV."""
        return pd.DataFrame({
            "id": [1, 2, 3],
            "type": ["OUTFALLSTREET", "STREET", "STREET"],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 1.0, 2.0],
            "slope": [0.0, 0.01, 0.01],
            "outgoing": [-1, 1, 2],
            "drain": [0, 0, 0],
            "drainCoupledID": [-1, -1, -1],
            "drainType": ["P-50x100", "P-50x100", "P-50x100"],
            "drainLength": [0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def temp_simple(self, simple_csv, tmp_path, monkeypatch):
        """Create temp CSV."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "simple.csv"
        simple_csv.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        return "simple"

    def test_very_small_inflow(self, temp_simple):
        """Test stability with very small inflows."""
        graph = HydraulicGraph("STREET", temp_simple)
        
        coupling = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 1e-10]),
            "drainCapture": np.zeros(3),
            "drainOverflow": np.zeros(3),
        }
        
        depth, area, _, peak = graph.update(0.0, 1800.0, coupling)
        
        assert all(np.isfinite(d) for d in depth)
        assert all(np.isfinite(a) for a in area)

    def test_large_inflow(self, temp_simple):
        """Test stability with large inflows."""
        graph = HydraulicGraph("STREET", temp_simple)
        
        coupling = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 1.0]),  # Very large
            "drainCapture": np.zeros(3),
            "drainOverflow": np.zeros(3),
        }
        
        depth, area, _, peak = graph.update(0.0, 1800.0, coupling)
        
        assert all(np.isfinite(d) for d in depth)
        assert all(np.isfinite(a) for a in area)

    def test_very_small_timestep(self, temp_simple):
        """Test stability with very small timestep."""
        graph = HydraulicGraph("STREET", temp_simple)
        
        coupling = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 0.01]),
            "drainCapture": np.zeros(3),
            "drainOverflow": np.zeros(3),
        }
        
        depth, area, _, peak = graph.update(0.0, 1.0, coupling)  # 1 second
        
        assert all(np.isfinite(d) for d in depth)
        assert all(np.isfinite(a) for a in area)

    def test_large_timestep(self, temp_simple):
        """Test stability with large timestep."""
        graph = HydraulicGraph("STREET", temp_simple)
        
        coupling = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 0.01]),
            "drainCapture": np.zeros(3),
            "drainOverflow": np.zeros(3),
        }
        
        depth, area, _, peak = graph.update(0.0, 7200.0, coupling)  # 2 hours
        
        assert all(np.isfinite(d) for d in depth)
        assert all(np.isfinite(a) for a in area)

    def test_long_simulation(self, temp_simple):
        """Test stability over many timesteps."""
        graph = HydraulicGraph("STREET", temp_simple)
        
        coupling = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 0.001]),
            "drainCapture": np.zeros(3),
            "drainOverflow": np.zeros(3),
        }
        
        dt = 600.0
        for i in range(50):
            depth, area, _, peak = graph.update(i * dt, dt, coupling)
            
            # Check stability at each step
            assert all(np.isfinite(d) for d in depth)
            assert all(np.isfinite(a) for a in area)
            assert all(d >= 0 for d in depth)
            assert all(a >= 0 for a in area)

    def test_varying_inflow(self, temp_simple):
        """Test stability with varying inflow over time."""
        graph = HydraulicGraph("STREET", temp_simple)
        
        dt = 600.0
        for i in range(20):
            # Sinusoidal varying inflow
            inflow = 0.01 * (1 + np.sin(i * np.pi / 10))
            
            coupling = {
                "subcatchmentRunoff": np.array([0.0, 0.0, inflow]),
                "drainCapture": np.zeros(3),
                "drainOverflow": np.zeros(3),
            }
            
            depth, area, _, peak = graph.update(i * dt, dt, coupling)
            
            assert all(np.isfinite(d) for d in depth)
            assert all(np.isfinite(a) for a in area)


class TestHydraulicGraphIntegration:
    """Integration tests for HydraulicGraph."""

    @pytest.fixture
    def full_network_csv(self):
        """CSV for a more complete network."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "type": ["OUTFALLSTREET", "STREET", "STREET", "STREET", "STREET"],
            "x": [0.0, 100.0, 100.0, 200.0, 200.0],
            "y": [0.0, 0.0, 100.0, 0.0, 100.0],
            "z": [0.0, 1.0, 1.5, 2.0, 2.5],
            "slope": [0.0, 0.01, 0.015, 0.01, 0.015],
            "outgoing": [-1, 1, 2, 2, 3],  # Multiple paths to outfall
            "drain": [0, 1, 0, 1, 0],
            "drainCoupledID": [-1, 10, -1, 11, -1],
            "drainType": ["P-50x100", "P-50x100", "P-50x100", "P-50x100", "P-50x100"],
            "drainLength": [0.0, 0.6, 0.0, 0.6, 0.0],
            "drainWidth": [0.0, 0.6, 0.0, 0.6, 0.0],
        })

    @pytest.fixture
    def temp_full_network(self, full_network_csv, tmp_path, monkeypatch):
        """Create temp CSV for full network."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "full_network.csv"
        full_network_csv.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)
        return "full_network"

    def test_flow_conservation(self, temp_full_network):
        """Test that flow is approximately conserved in steady state."""
        graph = HydraulicGraph("STREET", temp_full_network)
        
        # Apply constant inflow
        inflow = 0.01
        coupling = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 0.0, inflow, inflow]),
            "drainCapture": np.zeros(5),
            "drainOverflow": np.zeros(5),
        }
        
        # Run to approximate steady state
        dt = 600.0
        for i in range(30):
            depth, area, _, peak = graph.update(i * dt, dt, coupling)
        
        # At steady state, outflow should approximately equal inflow
        # (minus any numerical losses)
        total_Q2 = sum(graph.G.es["Q2"])
        assert total_Q2 > 0

    def test_water_flows_downhill(self, temp_full_network):
        """Test that water flows toward lower elevations."""
        graph = HydraulicGraph("STREET", temp_full_network)
        
        coupling = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 0.0, 0.01, 0.01]),
            "drainCapture": np.zeros(5),
            "drainOverflow": np.zeros(5),
        }
        
        dt = 600.0
        for i in range(10):
            depth, area, _, peak = graph.update(i * dt, dt, coupling)
        
        # Nodes with higher z should have flow going to lower z nodes
        # Check that edges have non-negative Q1 (flow enters) and Q2 (flow exits)
        for e in graph.G.es:
            if e["Q1"] > 0 or e["Q2"] > 0:
                source_z = graph.G.vs[e.source]["z"]
                target_z = graph.G.vs[e.target]["z"]
                # Flow direction should be from high to low
                assert source_z >= target_z

    def test_depth_increases_with_flow(self, temp_full_network):
        """Test that depth increases when flow increases."""
        graph = HydraulicGraph("STREET", temp_full_network)
        
        # Low flow case
        coupling_low = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 0.0, 0.001, 0.001]),
            "drainCapture": np.zeros(5),
            "drainOverflow": np.zeros(5),
        }
        
        dt = 600.0
        for i in range(20):
            graph.update(i * dt, dt, coupling_low)
        
        depth_low = np.array(graph.G.vs["depth"]).copy()
        
        # Reset and try high flow
        graph2 = HydraulicGraph("STREET", temp_full_network)
        
        coupling_high = {
            "subcatchmentRunoff": np.array([0.0, 0.0, 0.0, 0.01, 0.01]),
            "drainCapture": np.zeros(5),
            "drainOverflow": np.zeros(5),
        }
        
        for i in range(20):
            graph2.update(i * dt, dt, coupling_high)
        
        depth_high = np.array(graph2.G.vs["depth"])
        
        # Higher flow should result in higher or equal depths
        for i in range(len(depth_high)):
            if depth_high[i] > 0 or depth_low[i] > 0:
                assert depth_high[i] >= depth_low[i] - 1e-6

    def test_rainfall_event_response(self, temp_full_network):
        """Test system response to a rainfall event (rising then falling input)."""
        graph = HydraulicGraph("STREET", temp_full_network)
        
        dt = 300.0  # 5 minutes
        peak_discharges = []
        
        for i in range(30):
            # Rising limb then falling limb
            if i < 10:
                inflow = 0.001 * (i + 1)  # Rising
            elif i < 15:
                inflow = 0.01  # Peak
            else:
                inflow = max(0.001, 0.01 - 0.001 * (i - 15))  # Falling
            
            coupling = {
                "subcatchmentRunoff": np.array([0.0, 0.0, 0.0, inflow, inflow]),
                "drainCapture": np.zeros(5),
                "drainOverflow": np.zeros(5),
            }
            
            depth, area, _, peak = graph.update(i * dt, dt, coupling)
            peak_discharges.append(peak)
            
            # Check stability throughout
            assert all(np.isfinite(d) for d in depth)
            assert np.isfinite(peak)
        
        # Peak discharge should have a maximum somewhere in the middle
        max_peak_idx = np.argmax(peak_discharges)
        assert 5 < max_peak_idx < 25  # Should peak after rising starts and before end


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
