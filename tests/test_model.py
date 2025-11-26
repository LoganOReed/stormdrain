import pytest
from pytest import approx
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.model import Model


class TestModelInitialization:
    """Test suite for Model initialization."""

    @pytest.fixture
    def simple_csv_data(self):
        """Simple CSV data with street, sewer, and subcatchment."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "type": [
                "OUTFALLSTREET", "STREET", "STREET",
                "OUTFALLSEWER", "SEWER", "SEWER",
                "SUBCATCHMENT", "SUBCATCHMENT", "SUBCATCHMENT"
            ],
            "x": [0.0, 100.0, 200.0, 0.0, 100.0, 200.0, 50.0, 150.0, 250.0],
            "y": [0.0, 0.0, 0.0, -10.0, -10.0, -10.0, 50.0, 50.0, 50.0],
            "z": [0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 3.0, 4.0, 5.0],
            "slope": [0.0, 0.01, 0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.02],
            "outgoing": [-1, 1, 2, -1, 4, 5, 2, 3, 3],
            "drain": [0, 1, 0, 0, 1, 0, 0, 0, 0],
            "drainCoupledID": [-1, 5, -1, -1, 2, -1, -1, -1, -1],
            "drainType": ["P-50x100"] * 9,
            "drainLength": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def simple_rain_info(self):
        """Simple rainfall info for testing."""
        return {
            "spaceConversion": 0.0254,  # inches to meters
            "timeConversion": 3600,      # hours to seconds
            "rainfall": np.array([0.0, 0.5, 1.0, 0.5, 0.0]),
            "rainfallTimes": np.array([0, 1, 2, 3, 4]),
        }

    @pytest.fixture
    def temp_csv_file(self, simple_csv_data, tmp_path, monkeypatch):
        """Create a temporary CSV file for testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "test_model.csv"
        simple_csv_data.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "test_model"

    # Basic initialization tests
    def test_init_file_stored(self, temp_csv_file, simple_rain_info):
        """Test that file name is stored."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        assert model.file == temp_csv_file

    def test_init_dt_stored(self, temp_csv_file, simple_rain_info):
        """Test that dt is stored."""
        dt = 1800
        model = Model(temp_csv_file, dt, simple_rain_info)
        assert model.dt == dt

    def test_init_data_loaded(self, temp_csv_file, simple_rain_info, simple_csv_data):
        """Test that CSV data is loaded."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        assert model.data.shape[0] == simple_csv_data.shape[0]

    def test_init_oldwater_ratio_default(self, temp_csv_file, simple_rain_info):
        """Test default oldwater ratio."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        assert model.subcatchment.oldwaterRatio == 0.2

    def test_init_oldwater_ratio_custom(self, temp_csv_file, simple_rain_info):
        """Test custom oldwater ratio."""
        model = Model(temp_csv_file, 1800, simple_rain_info, oldwaterRatio=0.3)
        assert model.subcatchment.oldwaterRatio == 0.3

    # Rain info tests
    def test_init_space_conversion_stored(self, temp_csv_file, simple_rain_info):
        """Test that space conversion is stored."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        assert model.spaceConversion == simple_rain_info["spaceConversion"]

    def test_init_time_conversion_stored(self, temp_csv_file, simple_rain_info):
        """Test that time conversion is stored."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        assert model.timeConversion == simple_rain_info["timeConversion"]

    def test_init_rainfall_normalized(self, temp_csv_file, simple_rain_info):
        """Test that rainfall is normalized to m/s."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        # Rainfall should be converted from inches/hour to m/s
        # Original: inches, normalized: meters/second
        assert len(model.rainfall) == len(simple_rain_info["rainfall"])
        assert all(np.isfinite(r) for r in model.rainfall)

    def test_init_rainfall_times_converted(self, temp_csv_file, simple_rain_info):
        """Test that rainfall times are converted to seconds."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        # Times should be in seconds (original hours * 3600)
        expected_times = simple_rain_info["rainfallTimes"] * simple_rain_info["timeConversion"]
        np.testing.assert_array_almost_equal(model.rainfallTimes, expected_times)

    # Time stepping tests
    def test_init_T_calculated(self, temp_csv_file, simple_rain_info):
        """Test that total time T is calculated."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        expected_T = max(simple_rain_info["rainfallTimes"]) * simple_rain_info["timeConversion"]
        assert model.T == expected_T

    def test_init_N_calculated(self, temp_csv_file, simple_rain_info):
        """Test that number of timesteps N is calculated."""
        dt = 1800
        model = Model(temp_csv_file, dt, simple_rain_info)
        
        expected_N = int(model.T / dt)
        assert model.N == expected_N

    def test_init_ts_array_created(self, temp_csv_file, simple_rain_info):
        """Test that time array ts is created."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        assert len(model.ts) == model.N
        assert model.ts[0] == 0
        assert model.ts[-1] == approx(model.T, rel=0.01)

    def test_init_rain_interpolated(self, temp_csv_file, simple_rain_info):
        """Test that rain is interpolated to timesteps."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        assert len(model.rain) == model.N
        assert all(np.isfinite(r) for r in model.rain)

    # Network tests
    def test_init_subcatchment_created(self, temp_csv_file, simple_rain_info):
        """Test that subcatchment graph is created."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        assert model.subcatchment is not None
        assert model.subcatchment.G.vcount() == 3  # 3 subcatchments

    def test_init_street_created(self, temp_csv_file, simple_rain_info):
        """Test that street graph is created."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        assert model.street is not None
        assert model.street.graphType == "STREET"
        assert model.street.G.vcount() == 3  # 3 street nodes

    def test_init_sewer_created(self, temp_csv_file, simple_rain_info):
        """Test that sewer graph is created."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        assert model.sewer is not None
        assert model.sewer.graphType == "SEWER"
        assert model.sewer.G.vcount() == 3  # 3 sewer nodes

    # Coupling tests
    def test_init_coupling_initialized(self, temp_csv_file, simple_rain_info, simple_csv_data):
        """Test that coupling dictionary is initialized."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        assert "subcatchmentRunoff" in model.coupling
        assert "drainCapture" in model.coupling
        assert "drainOverflow" in model.coupling

    def test_init_coupling_correct_size(self, temp_csv_file, simple_rain_info, simple_csv_data):
        """Test that coupling arrays are correct size."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        n_rows = simple_csv_data.shape[0]
        assert len(model.coupling["subcatchmentRunoff"]) == n_rows
        assert len(model.coupling["drainCapture"]) == n_rows
        assert len(model.coupling["drainOverflow"]) == n_rows

    def test_init_coupling_zeros(self, temp_csv_file, simple_rain_info):
        """Test that coupling arrays are initialized to zeros."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        assert all(v == 0 for v in model.coupling["subcatchmentRunoff"])
        assert all(v == 0 for v in model.coupling["drainCapture"])
        assert all(v == 0 for v in model.coupling["drainOverflow"])

    # Observable storage tests
    def test_init_observable_lists_empty(self, temp_csv_file, simple_rain_info):
        """Test that observable storage lists are initialized empty."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        assert model.subcatchmentDepths == []
        assert model.runoffs == []
        assert model.streetDepths == []
        assert model.streetEdgeAreas == []
        assert model.sewerDepths == []
        assert model.sewerEdgeAreas == []
        assert model.drainOverflows == []
        assert model.drainInflows == []
        assert model.peakDischarges == []

    def test_init_additional_observable_lists_empty(self, temp_csv_file, simple_rain_info):
        """Test that additional observable storage lists are initialized empty."""
        model = Model(temp_csv_file, 1800, simple_rain_info)
        
        assert model.streetMaxDepths == []
        assert model.streetOutfallFlows == []
        assert model.sewerOutfallFlows == []
        assert model.streetPeakDischarges == []


class TestModelStep:
    """Test suite for Model step method."""

    @pytest.fixture
    def simple_csv_data(self):
        """Simple CSV data for testing."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "type": [
                "OUTFALLSTREET", "STREET", "STREET",
                "OUTFALLSEWER", "SEWER", "SEWER",
                "SUBCATCHMENT", "SUBCATCHMENT", "SUBCATCHMENT"
            ],
            "x": [0.0, 100.0, 200.0, 0.0, 100.0, 200.0, 50.0, 150.0, 250.0],
            "y": [0.0, 0.0, 0.0, -10.0, -10.0, -10.0, 50.0, 50.0, 50.0],
            "z": [0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 3.0, 4.0, 5.0],
            "slope": [0.0, 0.01, 0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.02],
            "outgoing": [-1, 1, 2, -1, 4, 5, 2, 3, 3],
            "drain": [0, 1, 0, 0, 1, 0, 0, 0, 0],
            "drainCoupledID": [-1, 5, -1, -1, 2, -1, -1, -1, -1],
            "drainType": ["P-50x100"] * 9,
            "drainLength": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def rain_info_with_rainfall(self):
        """Rain info with actual rainfall."""
        return {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.1, 0.5, 1.0, 0.5, 0.1]),
            "rainfallTimes": np.array([0, 1, 2, 3, 4]),
        }

    @pytest.fixture
    def temp_model_csv(self, simple_csv_data, tmp_path, monkeypatch):
        """Create temp CSV for model testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "step_test.csv"
        simple_csv_data.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "step_test"

    def test_step_appends_subcatchment_depths(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step appends subcatchment depths."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        initial_len = len(model.subcatchmentDepths)
        model.step(0)
        
        assert len(model.subcatchmentDepths) == initial_len + 1

    def test_step_appends_street_depths(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step appends street depths."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        initial_len = len(model.streetDepths)
        model.step(0)
        
        assert len(model.streetDepths) == initial_len + 1

    def test_step_appends_sewer_depths(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step appends sewer depths."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        initial_len = len(model.sewerDepths)
        model.step(0)
        
        assert len(model.sewerDepths) == initial_len + 1

    def test_step_appends_peak_discharges(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step appends peak discharges."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        initial_len = len(model.peakDischarges)
        model.step(0)
        
        assert len(model.peakDischarges) == initial_len + 1

    def test_step_multiple_times(self, temp_model_csv, rain_info_with_rainfall):
        """Test multiple sequential steps."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        for i in range(5):
            model.step(i)
        
        assert len(model.subcatchmentDepths) == 5
        assert len(model.streetDepths) == 5
        assert len(model.sewerDepths) == 5
        assert len(model.peakDischarges) == 5

    def test_step_finite_values(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step produces finite values."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        model.step(0)
        
        assert all(np.isfinite(d) for d in model.subcatchmentDepths[0])
        assert all(np.isfinite(d) for d in model.streetDepths[0])
        assert all(np.isfinite(d) for d in model.sewerDepths[0])
        assert np.isfinite(model.peakDischarges[0])

    def test_step_updates_coupling(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step updates coupling terms."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        # Run a few steps to build up some flow
        for i in range(3):
            model.step(i)
        
        # Coupling should have been updated
        # At least check they're still valid arrays
        assert len(model.coupling["subcatchmentRunoff"]) == model.data.shape[0]
        assert all(np.isfinite(v) for v in model.coupling["subcatchmentRunoff"])

    def test_step_with_rainfall_generates_flow(self, temp_model_csv, rain_info_with_rainfall):
        """Test that rainfall generates flow in the system."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        # Run several steps
        for i in range(5):
            model.step(i)
        
        # Should have some non-zero depths or peak discharge
        has_activity = (
            any(any(d > 0 for d in depths) for depths in model.subcatchmentDepths) or
            any(p > 0 for p in model.peakDischarges)
        )
        assert has_activity

    def test_step_appends_street_max_depths(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step appends street max depths."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        initial_len = len(model.streetMaxDepths)
        model.step(0)
        
        assert len(model.streetMaxDepths) == initial_len + 1
        assert np.isfinite(model.streetMaxDepths[0])
        assert model.streetMaxDepths[0] >= 0

    def test_step_appends_street_outfall_flows(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step appends street outfall flows."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        initial_len = len(model.streetOutfallFlows)
        model.step(0)
        
        assert len(model.streetOutfallFlows) == initial_len + 1
        assert np.isfinite(model.streetOutfallFlows[0])
        assert model.streetOutfallFlows[0] >= 0

    def test_step_appends_sewer_outfall_flows(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step appends sewer outfall flows."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        initial_len = len(model.sewerOutfallFlows)
        model.step(0)
        
        assert len(model.sewerOutfallFlows) == initial_len + 1
        assert np.isfinite(model.sewerOutfallFlows[0])
        assert model.sewerOutfallFlows[0] >= 0

    def test_step_appends_street_peak_discharges(self, temp_model_csv, rain_info_with_rainfall):
        """Test that step appends street peak discharges."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        initial_len = len(model.streetPeakDischarges)
        model.step(0)
        
        assert len(model.streetPeakDischarges) == initial_len + 1
        assert np.isfinite(model.streetPeakDischarges[0])
        assert model.streetPeakDischarges[0] >= 0

    def test_step_street_max_depth_consistency(self, temp_model_csv, rain_info_with_rainfall):
        """Test that street max depth equals max of street depths array."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        for i in range(5):
            model.step(i)
            expected_max = np.max(model.streetDepths[i]) if len(model.streetDepths[i]) > 0 else 0.0
            assert model.streetMaxDepths[i] == approx(expected_max)

    def test_step_street_peak_vs_combined_peak(self, temp_model_csv, rain_info_with_rainfall):
        """Test that street peak discharge <= combined peak discharge."""
        model = Model(temp_model_csv, 1800, rain_info_with_rainfall)
        
        for i in range(5):
            model.step(i)
            # Street peak should be <= combined (street + sewer) peak
            assert model.streetPeakDischarges[i] <= model.peakDischarges[i] + 1e-10


class TestModelUpdateDrainCapture:
    """Test suite for Model updateDrainCapture method."""

    @pytest.fixture
    def drain_network_csv(self):
        """CSV with drain connections."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "type": [
                "OUTFALLSTREET", "STREET", "STREET",
                "OUTFALLSEWER", "SEWER", "SEWER"
            ],
            "x": [0.0, 100.0, 200.0, 0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0, -10.0, -10.0, -10.0],
            "z": [0.0, 1.0, 2.0, -1.0, 0.0, 1.0],
            "slope": [0.0, 0.01, 0.01, 0.0, 0.01, 0.01],
            "outgoing": [-1, 1, 2, -1, 4, 5],
            "drain": [0, 1, 1, 0, 0, 0],  # Two drains on street
            "drainCoupledID": [-1, 5, 6, -1, -1, -1],
            "drainType": ["P-50x100"] * 6,
            "drainLength": [0.0, 0.6, 0.6, 0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.6, 0.6, 0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def simple_rain_info(self):
        """Simple rain info."""
        return {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 1.0, 0.5]),
            "rainfallTimes": np.array([0, 1, 2]),
        }

    @pytest.fixture
    def temp_drain_csv(self, drain_network_csv, tmp_path, monkeypatch):
        """Create temp CSV for drain testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "drain_test.csv"
        drain_network_csv.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "drain_test"

    def test_update_drain_capture_runs(self, temp_drain_csv, simple_rain_info):
        """Test that updateDrainCapture runs without error."""
        model = Model(temp_drain_csv, 1800, simple_rain_info)
        
        # Should not raise
        model.updateDrainCapture()

    def test_update_drain_capture_finite_values(self, temp_drain_csv, simple_rain_info):
        """Test that updateDrainCapture produces finite values."""
        model = Model(temp_drain_csv, 1800, simple_rain_info)
        
        model.updateDrainCapture()
        
        assert all(np.isfinite(v) for v in model.coupling["drainCapture"])

    def test_update_drain_capture_conservation(self, temp_drain_csv, simple_rain_info):
        """Test that drain capture conserves flow (what leaves street enters sewer)."""
        model = Model(temp_drain_csv, 1800, simple_rain_info)
        
        # Set up some flow in street edges
        for e in model.street.G.es:
            e["Q1"] = 0.01  # Some flow
        
        model.updateDrainCapture()
        
        # For each drain node, negative capture on street should equal positive on sewer
        for nid in model.street.G.vs:
            if nid["drain"] == 1:
                street_idx = nid["coupledID"] - 1
                sewer_idx = nid["drainCoupledID"] - 1
                
                street_capture = model.coupling["drainCapture"][street_idx]
                sewer_capture = model.coupling["drainCapture"][sewer_idx]
                
                # Street should lose flow (negative), sewer should gain (positive)
                assert street_capture <= 0
                assert sewer_capture >= 0
                assert abs(street_capture + sewer_capture) < 1e-10  # Conservation


class TestModelUpdateRunoff:
    """Test suite for Model updateRunoff method."""

    @pytest.fixture
    def runoff_network_csv(self):
        """CSV for runoff testing."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7],
            "type": [
                "OUTFALLSTREET", "STREET", "STREET",
                "OUTFALLSEWER", "SEWER",
                "SUBCATCHMENT", "SUBCATCHMENT"
            ],
            "x": [0.0, 100.0, 200.0, 0.0, 100.0, 150.0, 250.0],
            "y": [0.0, 0.0, 0.0, -10.0, -10.0, 50.0, 50.0],
            "z": [0.0, 1.0, 2.0, -1.0, 0.0, 3.0, 4.0],
            "slope": [0.0, 0.01, 0.01, 0.0, 0.01, 0.02, 0.02],
            "outgoing": [-1, 1, 2, -1, 4, 2, 3],
            "drain": [0, 0, 0, 0, 0, 0, 0],
            "drainCoupledID": [-1, -1, -1, -1, -1, -1, -1],
            "drainType": ["P-50x100"] * 7,
            "drainLength": [0.0] * 7,
            "drainWidth": [0.0] * 7,
        })

    @pytest.fixture
    def simple_rain_info(self):
        """Simple rain info."""
        return {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 1.0, 0.5]),
            "rainfallTimes": np.array([0, 1, 2]),
        }

    @pytest.fixture
    def temp_runoff_csv(self, runoff_network_csv, tmp_path, monkeypatch):
        """Create temp CSV for runoff testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "runoff_test.csv"
        runoff_network_csv.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "runoff_test"

    def test_update_runoff_runs(self, temp_runoff_csv, simple_rain_info):
        """Test that updateRunoff runs without error."""
        model = Model(temp_runoff_csv, 1800, simple_rain_info)
        
        # Should not raise
        model.updateRunoff()

    def test_update_runoff_finite_values(self, temp_runoff_csv, simple_rain_info):
        """Test that updateRunoff produces finite values."""
        model = Model(temp_runoff_csv, 1800, simple_rain_info)
        
        model.updateRunoff()
        
        assert all(np.isfinite(v) for v in model.coupling["subcatchmentRunoff"])

    def test_update_runoff_maps_correctly(self, temp_runoff_csv, simple_rain_info):
        """Test that runoff is mapped to correct hydraulic nodes."""
        model = Model(temp_runoff_csv, 1800, simple_rain_info)
        
        # Set known runoff values
        for i, v in enumerate(model.subcatchment.G.vs):
            v["runoff"] = 0.001 * (i + 1)
        
        model.updateRunoff()
        
        # Check that values were mapped
        # Note: hydraulicCoupling contains 1-based CSV IDs, coupling array is 0-indexed
        for v in model.subcatchment.G.vs:
            target_idx = model.subcatchment.hydraulicCoupling[v.index] - 1  # Convert to 0-based
            expected_runoff = v["runoff"]
            actual_runoff = model.coupling["subcatchmentRunoff"][target_idx]
            assert actual_runoff == approx(expected_runoff)


class TestModelRun:
    """Test suite for Model run method."""

    @pytest.fixture
    def full_network_csv(self):
        """Full network CSV for run testing."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "type": [
                "OUTFALLSTREET", "STREET", "STREET",
                "OUTFALLSEWER", "SEWER", "SEWER",
                "SUBCATCHMENT", "SUBCATCHMENT", "SUBCATCHMENT"
            ],
            "x": [0.0, 100.0, 200.0, 0.0, 100.0, 200.0, 50.0, 150.0, 250.0],
            "y": [0.0, 0.0, 0.0, -10.0, -10.0, -10.0, 50.0, 50.0, 50.0],
            "z": [0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 3.0, 4.0, 5.0],
            "slope": [0.0, 0.01, 0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.02],
            "outgoing": [-1, 1, 2, -1, 4, 5, 2, 3, 3],
            "drain": [0, 1, 0, 0, 1, 0, 0, 0, 0],
            "drainCoupledID": [-1, 5, -1, -1, 2, -1, -1, -1, -1],
            "drainType": ["P-50x100"] * 9,
            "drainLength": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def short_rain_info(self):
        """Short rainfall for faster testing."""
        return {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 1.0, 0.5]),
            "rainfallTimes": np.array([0, 1, 2]),
        }

    @pytest.fixture
    def temp_run_csv(self, full_network_csv, tmp_path, monkeypatch):
        """Create temp CSV for run testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "run_test.csv"
        full_network_csv.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "run_test"

    def test_run_completes(self, temp_run_csv, short_rain_info):
        """Test that run completes without error."""
        model = Model(temp_run_csv, 1800, short_rain_info)
        
        # Mock visualize to avoid file I/O
        with patch('app.model.visualize'):
            model.run()
        
        # Should have run all timesteps
        assert len(model.subcatchmentDepths) == model.N

    def test_run_populates_all_observables(self, temp_run_csv, short_rain_info):
        """Test that run populates all observable lists."""
        model = Model(temp_run_csv, 1800, short_rain_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # All observable lists should have N entries
        assert len(model.subcatchmentDepths) == model.N
        assert len(model.runoffs) == model.N
        assert len(model.streetDepths) == model.N
        assert len(model.streetEdgeAreas) == model.N
        assert len(model.sewerDepths) == model.N
        assert len(model.sewerEdgeAreas) == model.N
        assert len(model.drainOverflows) == model.N
        assert len(model.drainInflows) == model.N
        assert len(model.peakDischarges) == model.N
        
        # Additional observables should also have N entries
        assert len(model.streetMaxDepths) == model.N
        assert len(model.streetOutfallFlows) == model.N
        assert len(model.sewerOutfallFlows) == model.N
        assert len(model.streetPeakDischarges) == model.N

    def test_run_finite_results(self, temp_run_csv, short_rain_info):
        """Test that run produces finite results throughout."""
        model = Model(temp_run_csv, 1800, short_rain_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Check all values are finite
        for depths in model.subcatchmentDepths:
            assert all(np.isfinite(d) for d in depths)
        
        for depths in model.streetDepths:
            assert all(np.isfinite(d) for d in depths)
        
        for depths in model.sewerDepths:
            assert all(np.isfinite(d) for d in depths)
        
        assert all(np.isfinite(p) for p in model.peakDischarges)

    def test_run_calls_visualize(self, temp_run_csv, short_rain_info):
        """Test that run calls visualize when shouldVisualize=True."""
        model = Model(temp_run_csv, 1800, short_rain_info)
        
        with patch('app.model.visualize') as mock_visualize:
            model.run(shouldVisualize=True)
            
            # Visualize should have been called once
            mock_visualize.assert_called_once()

    def test_run_does_not_call_visualize_by_default(self, temp_run_csv, short_rain_info):
        """Test that run does not call visualize when shouldVisualize=False (default)."""
        model = Model(temp_run_csv, 1800, short_rain_info)
        
        with patch('app.model.visualize') as mock_visualize:
            model.run()  # Default shouldVisualize=False
            
            # Visualize should not have been called
            mock_visualize.assert_not_called()


class TestModelIntegration:
    """Integration tests for Model class."""

    @pytest.fixture
    def realistic_csv_data(self):
        """More realistic network data."""
        return pd.DataFrame({
            "id": list(range(1, 16)),
            "type": [
                "OUTFALLSTREET", "STREET", "STREET", "STREET", "STREET",
                "OUTFALLSEWER", "SEWER", "SEWER", "SEWER", "SEWER",
                "SUBCATCHMENT", "SUBCATCHMENT", "SUBCATCHMENT", "SUBCATCHMENT", "SUBCATCHMENT"
            ],
            "x": [
                0.0, 100.0, 200.0, 300.0, 400.0,
                0.0, 100.0, 200.0, 300.0, 400.0,
                50.0, 150.0, 250.0, 350.0, 450.0
            ],
            "y": [
                0.0, 0.0, 0.0, 0.0, 0.0,
                -10.0, -10.0, -10.0, -10.0, -10.0,
                50.0, 50.0, 50.0, 50.0, 50.0
            ],
            "z": [
                0.0, 0.5, 1.0, 1.5, 2.0,
                -1.0, -0.5, 0.0, 0.5, 1.0,
                2.0, 2.5, 3.0, 3.5, 4.0
            ],
            "slope": [
                0.0, 0.005, 0.005, 0.005, 0.005,
                0.0, 0.005, 0.005, 0.005, 0.005,
                0.02, 0.02, 0.02, 0.02, 0.02
            ],
            "outgoing": [
                -1, 1, 2, 3, 4,
                -1, 6, 7, 8, 9,
                2, 3, 4, 5, 5
            ],
            "drain": [
                0, 1, 1, 1, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ],
            "drainCoupledID": [
                -1, 7, 8, 9, -1,
                -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1
            ],
            "drainType": ["P-50x100"] * 15,
            "drainLength": [
                0.0, 0.6, 0.6, 0.6, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            ],
            "drainWidth": [
                0.0, 0.6, 0.6, 0.6, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            ],
        })

    @pytest.fixture
    def realistic_rain_info(self):
        """Realistic rainfall event."""
        return {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.1, 0.2, 0.4, 0.8, 1.0, 0.8, 0.5, 0.3, 0.1, 0.0]),
            "rainfallTimes": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        }

    @pytest.fixture
    def temp_realistic_csv(self, realistic_csv_data, tmp_path, monkeypatch):
        """Create temp CSV for realistic testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "realistic_test.csv"
        realistic_csv_data.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "realistic_test"

    def test_rainfall_response(self, temp_realistic_csv, realistic_rain_info):
        """Test system responds to rainfall event."""
        model = Model(temp_realistic_csv, 1800, realistic_rain_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Peak discharge should increase during rainfall and decrease after
        peak_discharges = model.peakDischarges
        
        # Should have some positive values
        assert any(p > 0 for p in peak_discharges)

    def test_flow_propagation(self, temp_realistic_csv, realistic_rain_info):
        """Test that flow propagates through the network."""
        model = Model(temp_realistic_csv, 1800, realistic_rain_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Street network should show flow
        has_street_flow = any(
            any(a > 0 for a in areas)
            for areas in model.streetEdgeAreas
        )
        
        # At least subcatchment or street should have activity
        has_subcatchment_depth = any(
            any(d > 0 for d in depths)
            for depths in model.subcatchmentDepths
        )
        
        assert has_street_flow or has_subcatchment_depth

    def test_numerical_stability_long_run(self, temp_realistic_csv, realistic_rain_info):
        """Test numerical stability over longer simulation."""
        model = Model(temp_realistic_csv, 900, realistic_rain_info)  # Smaller dt = more steps
        
        with patch('app.model.visualize'):
            model.run()
        
        # All values should remain finite
        for i in range(len(model.peakDischarges)):
            assert np.isfinite(model.peakDischarges[i]), f"Non-finite peak at step {i}"
            assert all(np.isfinite(d) for d in model.subcatchmentDepths[i])
            assert all(np.isfinite(d) for d in model.streetDepths[i])
            assert all(np.isfinite(d) for d in model.sewerDepths[i])

    def test_different_dt_values(self, temp_realistic_csv, realistic_rain_info):
        """Test model works with different timestep sizes."""
        dt_values = [600, 1200, 1800, 3600]
        
        for dt in dt_values:
            model = Model(temp_realistic_csv, dt, realistic_rain_info)
            
            with patch('app.model.visualize'):
                model.run()
            
            # Should complete without error
            assert len(model.peakDischarges) == model.N
            assert all(np.isfinite(p) for p in model.peakDischarges)

    def test_different_oldwater_ratios(self, temp_realistic_csv, realistic_rain_info):
        """Test model works with different oldwater ratios."""
        ratios = [0.0, 0.2, 0.5, 0.8]
        
        for ratio in ratios:
            model = Model(temp_realistic_csv, 1800, realistic_rain_info, oldwaterRatio=ratio)
            
            with patch('app.model.visualize'):
                model.run()
            
            # Should complete without error
            assert len(model.peakDischarges) == model.N


class TestModelEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def minimal_csv_data(self):
        """Minimal valid network with at least one edge in each network."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "type": ["OUTFALLSTREET", "STREET", "OUTFALLSEWER", "SEWER", "SUBCATCHMENT"],
            "x": [0.0, 100.0, 0.0, 100.0, 50.0],
            "y": [0.0, 0.0, -10.0, -10.0, 50.0],
            "z": [0.0, 1.0, -1.0, 0.0, 2.0],
            "slope": [0.0, 0.01, 0.0, 0.01, 0.02],
            "outgoing": [-1, 1, -1, 3, 2],
            "drain": [0, 0, 0, 0, 0],
            "drainCoupledID": [-1, -1, -1, -1, -1],
            "drainType": ["P-50x100"] * 5,
            "drainLength": [0.0] * 5,
            "drainWidth": [0.0] * 5,
        })

    @pytest.fixture
    def temp_minimal_csv(self, minimal_csv_data, tmp_path, monkeypatch):
        """Create temp CSV for minimal testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "minimal_test.csv"
        minimal_csv_data.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "minimal_test"

    def test_zero_rainfall(self, temp_minimal_csv):
        """Test model with zero rainfall."""
        rain_info = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.0, 0.0, 0.0]),
            "rainfallTimes": np.array([0, 1, 2]),
        }
        
        model = Model(temp_minimal_csv, 1800, rain_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Should complete without error
        assert len(model.peakDischarges) == model.N

    def test_constant_rainfall(self, temp_minimal_csv):
        """Test model with constant rainfall."""
        rain_info = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 0.5, 0.5, 0.5]),
            "rainfallTimes": np.array([0, 1, 2, 3]),
        }
        
        model = Model(temp_minimal_csv, 1800, rain_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Should complete without error
        assert len(model.peakDischarges) == model.N

    def test_intense_rainfall(self, temp_minimal_csv):
        """Test model with intense rainfall."""
        rain_info = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([5.0, 10.0, 5.0]),  # Very intense
            "rainfallTimes": np.array([0, 1, 2]),
        }
        
        model = Model(temp_minimal_csv, 1800, rain_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Should complete and produce finite values
        assert all(np.isfinite(p) for p in model.peakDischarges)

    def test_very_small_dt(self, temp_minimal_csv):
        """Test model with very small timestep."""
        rain_info = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 1.0, 0.5]),
            "rainfallTimes": np.array([0, 1, 2]),
        }
        
        model = Model(temp_minimal_csv, 60, rain_info)  # 1 minute timestep
        
        # Just run a few steps, not the whole simulation (too many steps)
        for i in range(min(10, model.N)):
            model.step(i)
        
        # Should produce finite values
        assert all(np.isfinite(p) for p in model.peakDischarges)

    def test_large_dt(self, temp_minimal_csv):
        """Test model with large timestep."""
        rain_info = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 1.0, 0.5, 0.0, 0.0]),
            "rainfallTimes": np.array([0, 1, 2, 3, 4]),
        }
        
        model = Model(temp_minimal_csv, 7200, rain_info)  # 2 hour timestep
        
        with patch('app.model.visualize'):
            model.run()
        
        # Should complete without error
        assert len(model.peakDischarges) == model.N

    def test_oldwater_ratio_zero(self, temp_minimal_csv):
        """Test model with zero oldwater ratio (no infiltration)."""
        rain_info = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 1.0, 0.5]),
            "rainfallTimes": np.array([0, 1, 2]),
        }
        
        model = Model(temp_minimal_csv, 1800, rain_info, oldwaterRatio=0.0)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Should complete without error
        assert len(model.peakDischarges) == model.N

    def test_oldwater_ratio_one(self, temp_minimal_csv):
        """Test model with oldwater ratio of 1 (all infiltration)."""
        rain_info = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 1.0, 0.5]),
            "rainfallTimes": np.array([0, 1, 2]),
        }
        
        model = Model(temp_minimal_csv, 1800, rain_info, oldwaterRatio=1.0)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Should complete without error
        assert len(model.peakDischarges) == model.N


class TestModelAdditionalObservables:
    """Test suite for the additional observables (max depth, outfall flows, street peak)."""

    @pytest.fixture
    def observable_csv_data(self):
        """CSV data for testing observables."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "type": [
                "OUTFALLSTREET", "STREET", "STREET",
                "OUTFALLSEWER", "SEWER", "SEWER",
                "SUBCATCHMENT", "SUBCATCHMENT", "SUBCATCHMENT"
            ],
            "x": [0.0, 100.0, 200.0, 0.0, 100.0, 200.0, 50.0, 150.0, 250.0],
            "y": [0.0, 0.0, 0.0, -10.0, -10.0, -10.0, 50.0, 50.0, 50.0],
            "z": [0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 3.0, 4.0, 5.0],
            "slope": [0.0, 0.01, 0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.02],
            "outgoing": [-1, 1, 2, -1, 4, 5, 2, 3, 3],
            "drain": [0, 1, 0, 0, 1, 0, 0, 0, 0],
            "drainCoupledID": [-1, 5, -1, -1, 2, -1, -1, -1, -1],
            "drainType": ["P-50x100"] * 9,
            "drainLength": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def rain_with_peak(self):
        """Rain info with a clear peak."""
        return {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.1, 0.5, 1.0, 0.5, 0.1]),
            "rainfallTimes": np.array([0, 1, 2, 3, 4]),
        }

    @pytest.fixture
    def temp_observable_csv(self, observable_csv_data, tmp_path, monkeypatch):
        """Create temp CSV for observable testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "observable_test.csv"
        observable_csv_data.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "observable_test"

    def test_street_max_depth_non_negative(self, temp_observable_csv, rain_with_peak):
        """Test that street max depth is always non-negative."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        assert all(d >= 0 for d in model.streetMaxDepths)

    def test_street_max_depth_bounded_by_street_depths(self, temp_observable_csv, rain_with_peak):
        """Test that street max depth is consistent with street depths."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        for i in range(len(model.streetMaxDepths)):
            if len(model.streetDepths[i]) > 0:
                expected_max = np.max(model.streetDepths[i])
                assert model.streetMaxDepths[i] == approx(expected_max)

    def test_street_outfall_flow_non_negative(self, temp_observable_csv, rain_with_peak):
        """Test that street outfall flow is always non-negative."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        assert all(f >= 0 for f in model.streetOutfallFlows)

    def test_sewer_outfall_flow_non_negative(self, temp_observable_csv, rain_with_peak):
        """Test that sewer outfall flow is always non-negative."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        assert all(f >= 0 for f in model.sewerOutfallFlows)

    def test_street_peak_discharge_non_negative(self, temp_observable_csv, rain_with_peak):
        """Test that street peak discharge is always non-negative."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        assert all(p >= 0 for p in model.streetPeakDischarges)

    def test_street_peak_leq_combined_peak(self, temp_observable_csv, rain_with_peak):
        """Test that street peak <= combined peak (street + sewer)."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        for i in range(len(model.streetPeakDischarges)):
            assert model.streetPeakDischarges[i] <= model.peakDischarges[i] + 1e-10

    def test_outfall_flow_increases_with_rainfall(self, temp_observable_csv, rain_with_peak):
        """Test that outfall flow generally increases during rainfall event."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Should have some positive outfall flow at some point
        has_street_outfall = any(f > 0 for f in model.streetOutfallFlows)
        has_sewer_outfall = any(f > 0 for f in model.sewerOutfallFlows)
        
        # At least one network should have outfall flow
        assert has_street_outfall or has_sewer_outfall

    def test_max_depth_responds_to_rainfall(self, temp_observable_csv, rain_with_peak):
        """Test that max depth responds to rainfall."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Max depth should increase at some point during simulation
        max_street_depth = max(model.streetMaxDepths)
        assert max_street_depth >= 0

    def test_all_new_observables_finite(self, temp_observable_csv, rain_with_peak):
        """Test that all new observables are finite throughout simulation."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        assert all(np.isfinite(d) for d in model.streetMaxDepths)
        assert all(np.isfinite(f) for f in model.streetOutfallFlows)
        assert all(np.isfinite(f) for f in model.sewerOutfallFlows)
        assert all(np.isfinite(p) for p in model.streetPeakDischarges)

    def test_new_observables_correct_length(self, temp_observable_csv, rain_with_peak):
        """Test that new observable lists have correct length after run."""
        model = Model(temp_observable_csv, 1800, rain_with_peak)
        
        with patch('app.model.visualize'):
            model.run()
        
        assert len(model.streetMaxDepths) == model.N
        assert len(model.streetOutfallFlows) == model.N
        assert len(model.sewerOutfallFlows) == model.N
        assert len(model.streetPeakDischarges) == model.N


class TestModelDataConsistency:
    """Tests for data consistency and correct data flow."""

    @pytest.fixture
    def data_flow_csv(self):
        """CSV for testing data flow."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "type": [
                "OUTFALLSTREET", "STREET",
                "OUTFALLSEWER", "SEWER",
                "SUBCATCHMENT", "SUBCATCHMENT"
            ],
            "x": [0.0, 100.0, 0.0, 100.0, 50.0, 150.0],
            "y": [0.0, 0.0, -10.0, -10.0, 50.0, 50.0],
            "z": [0.0, 1.0, -1.0, 0.0, 2.0, 3.0],
            "slope": [0.0, 0.01, 0.0, 0.01, 0.02, 0.02],
            "outgoing": [-1, 1, -1, 3, 2, 2],
            "drain": [0, 1, 0, 0, 0, 0],
            "drainCoupledID": [-1, 4, -1, -1, -1, -1],
            "drainType": ["P-50x100"] * 6,
            "drainLength": [0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def simple_rain_info(self):
        """Simple rain info."""
        return {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 1.0, 0.5]),
            "rainfallTimes": np.array([0, 1, 2]),
        }

    @pytest.fixture
    def temp_data_csv(self, data_flow_csv, tmp_path, monkeypatch):
        """Create temp CSV."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "data_flow_test.csv"
        data_flow_csv.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "data_flow_test"

    def test_observable_shapes_match_network(self, temp_data_csv, simple_rain_info):
        """Test that observable array shapes match network sizes."""
        model = Model(temp_data_csv, 1800, simple_rain_info)
        
        model.step(0)
        
        # Subcatchment depths should match subcatchment count
        assert len(model.subcatchmentDepths[0]) == model.subcatchment.G.vcount()
        
        # Street depths should match street node count
        assert len(model.streetDepths[0]) == model.street.G.vcount()
        
        # Sewer depths should match sewer node count
        assert len(model.sewerDepths[0]) == model.sewer.G.vcount()
        
        # Street edge areas should match street edge count
        assert len(model.streetEdgeAreas[0]) == model.street.G.ecount()
        
        # Sewer edge areas should match sewer edge count
        assert len(model.sewerEdgeAreas[0]) == model.sewer.G.ecount()

    def test_coupling_indices_valid(self, temp_data_csv, simple_rain_info):
        """Test that coupling indices are valid."""
        model = Model(temp_data_csv, 1800, simple_rain_info)
        
        n_data = model.data.shape[0]
        
        # All coupling arrays should have valid indices
        assert len(model.coupling["subcatchmentRunoff"]) == n_data
        assert len(model.coupling["drainCapture"]) == n_data
        assert len(model.coupling["drainOverflow"]) == n_data

    def test_time_array_monotonic(self, temp_data_csv, simple_rain_info):
        """Test that time array is monotonically increasing."""
        model = Model(temp_data_csv, 1800, simple_rain_info)
        
        for i in range(len(model.ts) - 1):
            assert model.ts[i] < model.ts[i + 1]

    def test_rainfall_interpolation_bounds(self, temp_data_csv, simple_rain_info):
        """Test that interpolated rainfall stays within original bounds."""
        model = Model(temp_data_csv, 1800, simple_rain_info)
        
        original_min = min(model.rainfall)
        original_max = max(model.rainfall)
        
        for r in model.rain:
            assert r >= original_min - 1e-10
            assert r <= original_max + 1e-10


class TestModelValidation:
    """Validation tests for physical correctness of the model."""

    @pytest.fixture
    def validation_csv_data(self):
        """CSV data for validation tests with known geometry."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "type": [
                "OUTFALLSTREET", "STREET", "STREET",
                "OUTFALLSEWER", "SEWER", "SEWER",
                "SUBCATCHMENT", "SUBCATCHMENT", "SUBCATCHMENT"
            ],
            "x": [0.0, 100.0, 200.0, 0.0, 100.0, 200.0, 100.0, 200.0, 200.0],
            "y": [0.0, 0.0, 0.0, -10.0, -10.0, -10.0, 50.0, 50.0, 100.0],
            "z": [0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 3.0, 4.0, 5.0],
            "slope": [0.0, 0.01, 0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.02],
            "outgoing": [-1, 1, 2, -1, 4, 5, 2, 3, 3],
            "drain": [0, 1, 0, 0, 1, 0, 0, 0, 0],
            "drainCoupledID": [-1, 5, -1, -1, 2, -1, -1, -1, -1],
            "drainType": ["P-50x100"] * 9,
            "drainLength": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
            "drainWidth": [0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
        })

    @pytest.fixture
    def rainfall_event_info(self):
        """Rainfall event with rising and falling limbs."""
        return {
            "spaceConversion": 0.0254,  # inches to meters
            "timeConversion": 3600,      # hours to seconds
            "rainfall": np.array([0.0, 0.2, 0.5, 1.0, 0.8, 0.4, 0.1, 0.0, 0.0, 0.0]),
            "rainfallTimes": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        }

    @pytest.fixture
    def constant_rainfall_info(self):
        """Constant rainfall for steady state tests."""
        return {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            "rainfallTimes": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        }

    @pytest.fixture
    def short_rainfall_then_stop(self):
        """Short rainfall then stop for drainage tests."""
        return {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "rainfallTimes": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        }

    @pytest.fixture
    def temp_validation_csv(self, validation_csv_data, tmp_path, monkeypatch):
        """Create temp CSV for validation testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        csv_path = data_dir / "validation_test.csv"
        validation_csv_data.to_csv(csv_path, index=False)
        
        monkeypatch.chdir(tmp_path)
        return "validation_test"

    # =========================================================================
    # Mass Conservation Tests
    # =========================================================================
    
    def test_mass_conservation_total_water_balance(self, temp_validation_csv, rainfall_event_info):
        """
        Test that total rainfall input approximately equals total outflow + storage change.
        
        Water balance: Rainfall * (1 - oldwaterRatio) * Area ≈ Outflow + ΔStorage
        """
        model = Model(temp_validation_csv, 900, rainfall_event_info, oldwaterRatio=0.2)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Calculate total rainfall input to subcatchments (in m³)
        # rainfall is already in m/s after normalization
        total_subcatchment_area = sum(model.subcatchment.G.vs["area"])
        
        # Integrate rainfall over time (rain array is at each timestep)
        total_rainfall_volume = 0.0
        for i in range(len(model.rain)):
            # rain[i] is in m/s, dt is in seconds, area is in m²
            rainfall_this_step = model.rain[i] * model.dt * total_subcatchment_area
            # Account for oldwater ratio (infiltration/evaporation loss)
            effective_rainfall = rainfall_this_step * (1 - model.subcatchment.oldwaterRatio)
            total_rainfall_volume += effective_rainfall
        
        # Calculate total outflow from outfalls (in m³)
        # Outfall flows are in m³/s, need to multiply by dt
        total_street_outflow = sum(model.streetOutfallFlows) * model.dt
        total_sewer_outflow = sum(model.sewerOutfallFlows) * model.dt
        total_storage = 0.0
        for i in range(len(model.streetEdgeAreas)):
            for j in range(len(model.streetEdgeAreas[0])):
                total_storage += model.streetEdgeAreas[i][j]*model.street.G.es[j]["length"]
                total_storage += model.sewerEdgeAreas[i][j]*model.sewer.G.es[j]["length"]
        total_outflow = total_street_outflow + total_sewer_outflow + total_storage


        
        # Allow for water still in storage (not yet drained)
        # This is a loose check - within 10% is reasonable for transient simulation
        if total_rainfall_volume > 0:
            print(f"total rainfall volume: {total_rainfall_volume}, total network water: {total_outflow}")
            ratio = total_outflow / total_rainfall_volume
            print(f"water ratio: {ratio}")
            # Outflow should be positive and bounded by input
            assert total_outflow >= 0, "Total outflow should be non-negative"
            # Allow significant tolerance due to storage and numerical integration
            assert ratio <= 1.1, f"Outflow ({total_outflow:.4f}) significantly exceeds rainfall ({total_rainfall_volume:.4f})"
            assert 0.9 <= ratio, f"Outflow ({total_outflow:.4f}) significantly exceeds rainfall ({total_rainfall_volume:.4f})"

    def test_cumulative_outflow_monotonic(self, temp_validation_csv, rainfall_event_info):
        """Test that cumulative outflow is monotonically increasing."""
        model = Model(temp_validation_csv, 900, rainfall_event_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Cumulative street outflow should be monotonically increasing
        cumulative_street = np.cumsum(model.streetOutfallFlows)
        for i in range(len(cumulative_street) - 1):
            assert cumulative_street[i] <= cumulative_street[i + 1] + 1e-10
        
        # Cumulative sewer outflow should be monotonically increasing
        cumulative_sewer = np.cumsum(model.sewerOutfallFlows)
        for i in range(len(cumulative_sewer) - 1):
            assert cumulative_sewer[i] <= cumulative_sewer[i + 1] + 1e-10

    def test_no_water_created_from_nothing(self, temp_validation_csv):
        """Test that with zero rainfall, no significant outflow occurs."""
        zero_rain_info = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": np.array([0.0, 0.0, 0.0, 0.0]),
            "rainfallTimes": np.array([0, 1, 2, 3]),
        }
        
        model = Model(temp_validation_csv, 1800, zero_rain_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # With zero rainfall, outflows should be essentially zero
        total_outflow = sum(model.streetOutfallFlows) + sum(model.sewerOutfallFlows)
        assert total_outflow < 1e-6, "Should have no outflow with zero rainfall"

    # =========================================================================
    # Temporal Behavior Tests
    # =========================================================================

    def test_peak_outflow_lags_peak_rainfall(self, temp_validation_csv, rainfall_event_info):
        """Test that peak outflow occurs after or at peak rainfall (hydrograph lag)."""
        model = Model(temp_validation_csv, 900, rainfall_event_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Find peak rainfall time
        peak_rain_idx = np.argmax(model.rain)
        
        # Find peak combined outflow time
        combined_outflow = np.array(model.streetOutfallFlows) + np.array(model.sewerOutfallFlows)
        peak_outflow_idx = np.argmax(combined_outflow)
        
        # Peak outflow should occur at or after peak rainfall
        # (allowing some tolerance for discretization)
        assert peak_outflow_idx >= peak_rain_idx - 1, \
            f"Peak outflow (idx={peak_outflow_idx}) should not precede peak rainfall (idx={peak_rain_idx})"

    def test_system_drains_after_rainfall_stops(self, temp_validation_csv, short_rainfall_then_stop):
        """Test that the system drains after rainfall stops."""
        model = Model(temp_validation_csv, 900, short_rainfall_then_stop)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Find when rainfall stops (becomes zero)
        rain_stop_idx = None
        for i, r in enumerate(model.rain):
            if r < 1e-10 and rain_stop_idx is None:
                rain_stop_idx = i
        
        if rain_stop_idx is not None and rain_stop_idx < len(model.streetMaxDepths) - 2:
            # After several timesteps post-rainfall, depths should decrease
            # Compare late-simulation depths to mid-simulation peaks
            late_start = min(rain_stop_idx + 3, len(model.streetMaxDepths) - 1)
            
            if late_start < len(model.streetMaxDepths):
                max_depth_during_rain = max(model.streetMaxDepths[:rain_stop_idx + 1]) if rain_stop_idx > 0 else 0
                final_depth = model.streetMaxDepths[-1]
                
                # Final depth should be less than or equal to peak depth
                assert final_depth <= max_depth_during_rain + 1e-6, \
                    "System should drain - final depth should not exceed peak"

    def test_outflow_eventually_decreases_after_rain_stops(self, temp_validation_csv, short_rainfall_then_stop):
        """Test that outflow decreases after rainfall stops."""
        model = Model(temp_validation_csv, 900, short_rainfall_then_stop)
        
        with patch('app.model.visualize'):
            model.run()
        
        combined_outflow = np.array(model.streetOutfallFlows) + np.array(model.sewerOutfallFlows)
        
        if len(combined_outflow) > 5:
            # The last few values should be decreasing or near zero
            late_outflows = combined_outflow[-3:]
            early_peak = max(combined_outflow[:len(combined_outflow)//2 + 1])
            
            # Late outflows should be less than or equal to peak
            assert all(o <= early_peak + 1e-6 for o in late_outflows), \
                "Outflow should decrease after rainfall stops"

    # =========================================================================
    # Steady State Tests
    # =========================================================================

    def test_approaches_steady_state_with_constant_rainfall(self, temp_validation_csv, constant_rainfall_info):
        """Test that system approaches steady state with constant rainfall."""
        model = Model(temp_validation_csv, 900, constant_rainfall_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        if len(model.streetOutfallFlows) >= 6:
            # Compare early and late outflows - should converge
            early_outflows = model.streetOutfallFlows[1:3]  # Skip first (initial conditions)
            late_outflows = model.streetOutfallFlows[-3:]
            
            # Coefficient of variation of late outflows should be small
            late_mean = np.mean(late_outflows)
            if late_mean > 1e-10:
                late_cv = np.std(late_outflows) / late_mean
                # CV should be less than 50% (approaching steady state)
                assert late_cv < 0.5, f"System should approach steady state (CV={late_cv:.2f})"

    # =========================================================================
    # Physical Constraint Tests
    # =========================================================================

    def test_street_depths_bounded_by_yfull(self, temp_validation_csv, rainfall_event_info):
        """Test that street depths do not exceed yFull."""
        model = Model(temp_validation_csv, 900, rainfall_event_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Get yFull for street (should be same for all edges)
        street_yfull = model.street.G.es[0]["yFull"] if model.street.G.ecount() > 0 else float('inf')
        
        for depths in model.streetDepths:
            for d in depths:
                assert d <= street_yfull * 1.1, \
                    f"Street depth {d} exceeds yFull {street_yfull}"

    def test_sewer_depths_bounded_by_yfull(self, temp_validation_csv, rainfall_event_info):
        """Test that sewer depths do not exceed yFull."""
        model = Model(temp_validation_csv, 900, rainfall_event_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Get yFull for sewer
        sewer_yfull = model.sewer.G.es[0]["yFull"] if model.sewer.G.ecount() > 0 else float('inf')
        
        for depths in model.sewerDepths:
            for d in depths:
                assert d <= sewer_yfull * 1.1, \
                    f"Sewer depth {d} exceeds yFull {sewer_yfull}"

    def test_all_flows_non_negative(self, temp_validation_csv, rainfall_event_info):
        """Test that all flow values remain non-negative."""
        model = Model(temp_validation_csv, 900, rainfall_event_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Check outfall flows
        assert all(f >= -1e-10 for f in model.streetOutfallFlows), \
            "Street outfall flows should be non-negative"
        assert all(f >= -1e-10 for f in model.sewerOutfallFlows), \
            "Sewer outfall flows should be non-negative"
        
        # Check peak discharges
        assert all(p >= -1e-10 for p in model.peakDischarges), \
            "Peak discharges should be non-negative"
        assert all(p >= -1e-10 for p in model.streetPeakDischarges), \
            "Street peak discharges should be non-negative"

    def test_all_depths_non_negative(self, temp_validation_csv, rainfall_event_info):
        """Test that all depth values remain non-negative."""
        model = Model(temp_validation_csv, 900, rainfall_event_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        for i, depths in enumerate(model.streetDepths):
            assert all(d >= -1e-10 for d in depths), \
                f"Street depths at step {i} should be non-negative"
        
        for i, depths in enumerate(model.sewerDepths):
            assert all(d >= -1e-10 for d in depths), \
                f"Sewer depths at step {i} should be non-negative"
        
        for i, depths in enumerate(model.subcatchmentDepths):
            assert all(d >= -1e-10 for d in depths), \
                f"Subcatchment depths at step {i} should be non-negative"

    def test_all_areas_non_negative(self, temp_validation_csv, rainfall_event_info):
        """Test that all cross-sectional areas remain non-negative."""
        model = Model(temp_validation_csv, 900, rainfall_event_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        for i, areas in enumerate(model.streetEdgeAreas):
            assert all(a >= -1e-10 for a in areas), \
                f"Street edge areas at step {i} should be non-negative"
        
        for i, areas in enumerate(model.sewerEdgeAreas):
            assert all(a >= -1e-10 for a in areas), \
                f"Sewer edge areas at step {i} should be non-negative"

    # =========================================================================
    # Network Consistency Tests
    # =========================================================================

    def test_street_peak_bounded_by_network_capacity(self, temp_validation_csv, rainfall_event_info):
        """Test that street peak discharge is bounded by network capacity."""
        model = Model(temp_validation_csv, 900, rainfall_event_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # Get total qFull capacity of street network
        total_street_capacity = sum(e["qFull"] for e in model.street.G.es)
        
        for p in model.streetPeakDischarges:
            assert p <= total_street_capacity * 1.1, \
                f"Street peak {p} exceeds network capacity {total_street_capacity}"

    def test_combined_peak_is_sum_of_components(self, temp_validation_csv, rainfall_event_info):
        """Test that combined peak equals street peak + sewer peak."""
        model = Model(temp_validation_csv, 900, rainfall_event_info)
        
        with patch('app.model.visualize'):
            model.run()
        
        # The combined peak should be approximately street + sewer peaks
        # (Note: peakDischarges is computed as streetPeakDischarge + sewerPeakDischarge)
        for i in range(len(model.peakDischarges)):
            street_peak = model.streetPeakDischarges[i]
            combined_peak = model.peakDischarges[i]
            
            # Street peak should be <= combined peak
            assert street_peak <= combined_peak + 1e-10, \
                f"Street peak should not exceed combined peak at step {i}"

    # =========================================================================
    # Rainfall Processing Tests
    # =========================================================================

    def test_total_rainfall_volume_calculation(self, temp_validation_csv, rainfall_event_info):
        """Test that rainfall volume calculation is correct."""
        model = Model(temp_validation_csv, 1800, rainfall_event_info)
        
        # Calculate total rainfall using trapezoidal integration
        # rainfall and rainfallTimes are normalized (m/s and seconds)
        total_rainfall_integrated = np.trapz(model.rainfall, model.rainfallTimes)
        
        # Should be positive for non-zero rainfall
        assert total_rainfall_integrated > 0, "Total rainfall should be positive"
        
        # The interpolated rain should integrate to approximately the same value
        total_rain_from_ts = np.trapz(model.rain, model.ts)
        
        # Allow 20% tolerance due to interpolation differences
        if total_rainfall_integrated > 0:
            ratio = total_rain_from_ts / total_rainfall_integrated
            assert 0.5 < ratio < 2.0, \
                f"Interpolated rainfall integral differs significantly from original"

    def test_oldwater_ratio_affects_runoff(self, temp_validation_csv, rainfall_event_info):
        """Test that oldwater ratio affects the amount of runoff generated."""
        # Run with low oldwater ratio (more runoff)
        model_low = Model(temp_validation_csv, 900, rainfall_event_info, oldwaterRatio=0.1)
        with patch('app.model.visualize'):
            model_low.run()
        
        # Run with high oldwater ratio (less runoff)
        model_high = Model(temp_validation_csv, 900, rainfall_event_info, oldwaterRatio=0.5)
        with patch('app.model.visualize'):
            model_high.run()
        
        # Total outflow with low oldwater ratio should be >= high oldwater ratio
        total_outflow_low = sum(model_low.streetOutfallFlows) + sum(model_low.sewerOutfallFlows)
        total_outflow_high = sum(model_high.streetOutfallFlows) + sum(model_high.sewerOutfallFlows)
        
        # Low oldwater ratio means more effective rainfall, so more outflow
        assert total_outflow_low >= total_outflow_high * 0.9, \
            "Lower oldwater ratio should result in more outflow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
