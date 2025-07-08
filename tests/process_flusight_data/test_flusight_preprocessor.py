import pytest
import pandas as pd
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / 'scripts'))

from scripts.process_flusight_data import FluSightPreprocessor, NpEncoder


@pytest.fixture
def fake_filesystem(tmp_path: Path) -> Path:
    """Creates a fake FluSight-forecast-hub directory structure in a temporary directory."""
    base_path = tmp_path / "FluSight-forecast-hub"
    
    model_output_path = base_path / "model-output"
    target_data_path = base_path / "target-data"
    aux_data_path = base_path / "auxiliary-data"
    model_output_path.mkdir(parents=True)
    target_data_path.mkdir(parents=True)
    aux_data_path.mkdir(parents=True)

    locations_df = pd.DataFrame([
        {"location": "06", "abbreviation": "CA", "location_name": "California", "population": 39237836},
        {"location": "48", "abbreviation": "TX", "location_name": "Texas", "population": 29145505},
        {"location": "US", "abbreviation": "US", "location_name": "United States", "population": 331893745} 
    ])
    locations_df.to_csv(aux_data_path / "locations.csv", index=False)

    target_df = pd.DataFrame([
        {"date": "2023-10-14", "location": "06", "value": 100, "weekly_rate": 0.25},
        {"date": "2023-10-21", "location": "06", "value": 110, "weekly_rate": 0.28},
        {"date": "2023-10-14", "location": "US", "value": 1000, "weekly_rate": 0.3} 
    ])
    target_df.to_csv(target_data_path / "target-hospital-admissions.csv", index=False)

    model1_path = model_output_path / "UNC_IDD-influpaint"
    model1_path.mkdir()
    model1_df = pd.DataFrame({
        'reference_date': ['2023-10-09', '2023-10-09'],
        'target': ['1 wk ahead inc hosp', '1 wk ahead inc hosp'],
        'target_end_date': ['2023-10-14', '2023-10-14'],
        'horizon': [1, 1],
        'location': ['06', '06'],
        'output_type': ['quantile', 'quantile'],
        'output_type_id': [0.25, 0.75],
        'value': [90, 110]
    })
    model1_df.to_csv(model1_path / "2023-10-09-UNC_IDD-influpaint.csv", index=False)

    model2_path = model_output_path / "FluSight-ensemble"
    model2_path.mkdir()

    model3_path = model_output_path / "Another-Model"
    model3_path.mkdir() # This model should be ignored in demo mode
    model3_df = model1_df.copy()
    model3_df['location'] = '48'
    model3_df.to_csv(model3_path / "2023-10-09-Another-Model.csv", index=False)
    
    return base_path

@pytest.fixture
def preprocessor_instance(fake_filesystem: Path, tmp_path: Path) -> FluSightPreprocessor:
    """Provides a preprocessor instance pointing to the fake filesystem."""
    output_path = tmp_path / "processed_output"
    return FluSightPreprocessor(base_path=str(fake_filesystem), output_path=str(output_path))

@pytest.fixture
def mock_thread_pool(monkeypatch):
    """Disables multithreading for deterministic testing."""
    from concurrent.futures import ThreadPoolExecutor
    
    class SerialExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers=None, *args, **kwargs):
            super().__init__(max_workers=1) 
        
        def map(self, fn, *iterables, timeout=None, chunksize=1):
            return (fn(i) for i in iterables[0])

    monkeypatch.setattr("scripts.process_flusight_data.ThreadPoolExecutor", SerialExecutor)


class TestFluSightPreprocessor:

    def test_init_success(self, fake_filesystem, tmp_path):
        """Tests successful initialization and file discovery."""
        output_path = tmp_path / "output"
        preprocessor = FluSightPreprocessor(str(fake_filesystem), str(output_path))
        assert preprocessor.base_path.exists()
        assert output_path.exists()
        assert "UNC_IDD-influpaint" in preprocessor.model_files
        assert len(preprocessor.model_files["UNC_IDD-influpaint"]) == 1

    def test_init_path_not_found(self, tmp_path):
        """Tests that init fails if a required path is missing."""
        with pytest.raises(ValueError, match="base_path does not exist"):
            FluSightPreprocessor(str(tmp_path / "nonexistent"), str(tmp_path / "output"))

    def test_load_locations(self, preprocessor_instance):
        """Tests loading and caching of locations data."""
        locations_df = preprocessor_instance.load_locations()
        assert isinstance(locations_df, pd.DataFrame)
        assert "California" in locations_df["location_name"].values
        
        preprocessor_instance.locations_path = Path("invalid_path") # Invalidate path
        locations_df_cached = preprocessor_instance.load_locations() # Should return from cache
        assert locations_df is locations_df_cached

    def test_load_ground_truth(self, preprocessor_instance):
        """Tests loading and processing of ground truth data."""
        ground_truth = preprocessor_instance.load_ground_truth()
        assert "06" in ground_truth 
        assert ground_truth["06"]["dates"] == ["2023-10-14", "2023-10-21"]
        assert ground_truth["06"]["values"] == [100, 110]

    def test_read_model_outputs(self, preprocessor_instance, mock_thread_pool):
        """Tests reading and deep merging of all model forecast files."""
        forecast_data = preprocessor_instance.read_model_outputs()
        
        # Check structure for the CA forecast from the UNC model
        ca_forecast = forecast_data["06"]["2023-10-09"]["1 wk ahead inc hosp"]["UNC_IDD-influpaint"]
        assert ca_forecast["type"] == "quantile"
        assert "1" in ca_forecast["predictions"]
        assert ca_forecast["predictions"]["1"]["quantiles"] == [0.25, 0.75]

    def test_create_visualization_payloads(self, preprocessor_instance, mock_thread_pool):
        """Tests the end-to-end creation of JSON payloads."""
        preprocessor_instance.create_visualization_payloads()

        # Check that metadata.json was created and is valid
        metadata_path = preprocessor_instance.output_path / "flusight" / "metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert "UNC_IDD-influpaint" in metadata["models"]
        assert "Another-Model" in metadata["models"]
        assert len(metadata["locations"]) == 3 

        # Check that a location payload was created and is valid
        ca_payload_path = preprocessor_instance.output_path / "flusight" / "CA_flusight.json"
        assert ca_payload_path.exists()
        with open(ca_payload_path) as f:
            ca_payload = json.load(f)

        assert ca_payload["metadata"]["location_name"] == "California"
        assert ca_payload["ground_truth"]["values"][0] == 100
        assert "UNC_IDD-influpaint" in ca_payload["forecasts"]["2023-10-09"]["1 wk ahead inc hosp"]
        assert "UNC_IDD-influpaint" in ca_payload["available_models"]
        
    def test_demo_mode(self, fake_filesystem, tmp_path, mock_thread_pool):
        """Tests that demo mode correctly filters models."""
        output_path = tmp_path / "demo_output"
        preprocessor = FluSightPreprocessor(str(fake_filesystem), str(output_path), demo_mode=True)
        preprocessor.create_visualization_payloads()

        metadata_path = output_path / "flusight" / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "Another-Model" not in metadata["models"]
        assert "UNC_IDD-influpaint" in metadata["models"]