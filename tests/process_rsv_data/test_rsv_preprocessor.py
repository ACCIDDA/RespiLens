import pytest
import pandas as pd
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / 'scripts'))

from scripts.process_rsv_data import RSVPreprocessor


@pytest.fixture
def fake_filesystem(tmp_path: Path) -> Path:
    """Creates a fake rsv-forecast-hub directory structure in a temporary directory."""
    base_path = tmp_path / "rsv-forecast-hub"
    
    model_output_path = base_path / "model-output"
    target_data_path = base_path / "target-data"
    aux_data_path = base_path / "auxiliary-data" / "location_census"
    model_output_path.mkdir(parents=True)
    target_data_path.mkdir(parents=True)
    aux_data_path.mkdir(parents=True)

    locations_df = pd.DataFrame([
        {"location": "06", "abbreviation": "CA", "location_name": "California", "population": 39237836},
    ])
    locations_df.to_csv(aux_data_path / "locations.csv", index=False)

    target_df_old = pd.DataFrame({"date": ["2023-10-07"], "location": ["06"], "age_group": ["0-0.49"], "target": ["inc hosp"], "value": [1]})
    target_df_old.to_csv(target_data_path / "2023-10-10_rsvnet_hospitalization.csv", index=False)
    
    target_df_new = pd.DataFrame([
        {"date": "2023-10-14", "location": "06", "age_group": "0-0.49", "target": "inc hosp", "value": 10},
        {"date": "2023-10-14", "location": "06", "age_group": "0.5-0.99", "target": "inc hosp", "value": 5},
    ])
    target_df_new.to_csv(target_data_path / "2023-10-17_rsvnet_hospitalization.csv", index=False)

    model1_path = model_output_path / "Some-Model"
    model1_path.mkdir()
    model1_df = pd.DataFrame({
        'origin_date': [pd.Timestamp("2023-10-09")],
        'target': ['inc hosp'],
        'horizon': [1],
        'location': ['06'],
        'age_group': ['0-0.99'],
        'output_type': ['quantile'],
        'output_type_id': [0.5],
        'value': [16]
    })
    model1_df.to_parquet(model1_path / "2023-10-09-Some-Model.parquet")
    
    return base_path

@pytest.fixture
def preprocessor_instance(fake_filesystem: Path, tmp_path: Path) -> RSVPreprocessor:
    """Provides a preprocessor instance pointing to the fake filesystem."""
    output_path = tmp_path / "processed_output"
    return RSVPreprocessor(base_path=str(fake_filesystem), output_path=str(output_path))

@pytest.fixture
def mock_thread_pool(monkeypatch):
    """Disables multithreading for deterministic testing."""
    from concurrent.futures import ThreadPoolExecutor
    
    class SerialExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers=None, *args, **kwargs):
            super().__init__(max_workers=1)
        
        def map(self, fn, *iterables, timeout=None, chunksize=1):
            return (fn(i) for i in iterables[0])

    monkeypatch.setattr("scripts.process_rsv_data.ThreadPoolExecutor", SerialExecutor)


class TestRSVPreprocessor:

    def test_init_success(self, fake_filesystem, tmp_path):
        """Tests successful initialization and that the LATEST target file is chosen."""
        output_path = tmp_path / "output"
        preprocessor = RSVPreprocessor(str(fake_filesystem), str(output_path))
        assert "2023-10-17_rsvnet_hospitalization.csv" in str(preprocessor.target_data_path)
        assert "Some-Model" in preprocessor.model_files

    def test_init_no_target_data(self, tmp_path):
        """Tests that init fails if no target data files are found."""
        base_path = tmp_path / "empty-hub"
        base_path.mkdir()
        (base_path / "model-output").mkdir()
        (base_path / "target-data").mkdir()
        (base_path / "auxiliary-data/location_census").mkdir(parents=True)
        (base_path / "auxiliary-data/location_census/locations.csv").touch()


        with pytest.raises(ValueError, match="No target data files found"):
            RSVPreprocessor(str(base_path), str(tmp_path / "output"))

    def test_load_ground_truth_aggregation(self, preprocessor_instance):
        """Tests that ground truth data is correctly aggregated by age group."""
        ground_truth = preprocessor_instance.load_ground_truth()
        assert "06" in ground_truth
        
        agg_group = ground_truth["06"]["0-0.99"]
        assert agg_group["dates"] == ["2023-10-14"]
        assert agg_group["values"] == [15]

    def test_read_model_outputs(self, preprocessor_instance, mock_thread_pool):
        """Tests reading of parquet model files and creation of the nested data structure."""
        forecast_data = preprocessor_instance.read_model_outputs()
        
        assert "06" in forecast_data
        assert "2023-10-09" in forecast_data["06"]
        assert "0-0.99" in forecast_data["06"]["2023-10-09"]
        
        model_forecast = forecast_data["06"]["2023-10-09"]["0-0.99"]["inc hosp"]["Some-Model"]
        assert model_forecast["type"] == "quantile"
        assert model_forecast["predictions"]["1"]["values"] == [16]

    def test_create_visualization_payloads(self, preprocessor_instance, mock_thread_pool):
        """Tests the end-to-end creation of the final JSON payloads."""
        preprocessor_instance.create_visualization_payloads()

        metadata_path = preprocessor_instance.output_path / "rsv" / "metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert "Some-Model" in metadata["models"]
        assert "0-0.99" in metadata["age_groups"]

        ca_payload_path = preprocessor_instance.output_path / "rsv" / "CA_rsv.json"
        assert ca_payload_path.exists()
        with open(ca_payload_path) as f:
            payload = json.load(f)

        assert payload["metadata"]["location_name"] == "California"
        assert payload["ground_truth"]["0-0.99"]["values"] == [15]
        assert "Some-Model" in payload["forecasts"]["2023-10-09"]["0-0.99"]["inc hosp"]