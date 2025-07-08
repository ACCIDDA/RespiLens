import pytest
import pandas as pd
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / 'scripts'))

from scripts.process_nhsn_data import NHSNDataDownloader

# -- Fixtures --

@pytest.fixture
def fake_locations_csv(tmp_path: Path) -> Path:
    """Creates a fake locations.csv file and returns its path."""
    locations_path = tmp_path / "locations.csv"
    locations_df = pd.DataFrame([
        {"location": "06", "abbreviation": "CA", "location_name": "California", "population": 39237836},
        {"location": "48", "abbreviation": "TX", "location_name": "Texas", "population": 29145505},
        {"location": "US", "abbreviation": "US", "location_name": "United States", "population": 331893745}
    ])
    locations_df.to_csv(locations_path, index=False)
    return locations_path

@pytest.fixture
def downloader_instance(tmp_path: Path, fake_locations_csv: Path) -> NHSNDataDownloader:
    """Provides an instance of NHSNDataDownloader with mocked paths."""
    return NHSNDataDownloader(output_path=str(tmp_path), locations_path=str(fake_locations_csv))

@pytest.fixture
def mock_requests_get(monkeypatch):
    """Mocks requests.get to return fake NHSN data and simulate pagination."""
    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code
        def json(self):
            return self._json_data
        def raise_for_status(self):
            if self.status_code != 200:
                raise Exception(f"HTTP Error {self.status_code}")

    def mock_get(url, params=None):
        offset = params.get("$offset", 0)
        if offset > 0: # Simulate end of pagination
            return MockResponse([])

        if "ua7e-t2fy" in url: # Official data
            return MockResponse([{"jurisdiction": "CA", "weekendingdate": "2023-01-07T00:00:00.000", "totalconfrsvnewadm": 100}])
        if "mpgq-jmmr" in url: # Preliminary data
            return MockResponse([{"jurisdiction": "TX", "weekendingdate": "2023-01-14T00:00:00.000", "totalconfrsvnewadm": 50}])
        return MockResponse(None, 404)

    monkeypatch.setattr("requests.get", mock_get)

@pytest.fixture
def sample_downloaded_df() -> pd.DataFrame:
    """Provides a sample DataFrame as if it were downloaded."""
    return pd.DataFrame([
        {"jurisdiction": "CA", "weekendingdate": "2023-01-07", "totalconfrsvnewadm": 100, "_type": "official"},
        {"jurisdiction": "TX", "weekendingdate": "2023-01-14", "totalconfrsvnewadm": 50, "_type": "preliminary"},
        {"jurisdiction": "Region 1", "weekendingdate": "2023-01-14", "totalconfrsvnewadm": 5, "_type": "official"} # Should be filtered
    ])

# -- Test Class --

class TestNHSNDataDownloader:

    def test_init(self, tmp_path: Path):
        """Tests that the downloader initializes and creates the output directory."""
        output_path = tmp_path / "test_output"
        assert not output_path.exists()
        downloader = NHSNDataDownloader(output_path=str(output_path))
        assert downloader.output_path == output_path
        assert output_path.exists()

    def test_download_data(self, downloader_instance, mock_requests_get):
        """Tests the end-to-end download and concatenation logic."""
        df = downloader_instance.download_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "official" in df["_type"].values
        assert "preliminary" in df["_type"].values
        assert "CA" in df["jurisdiction"].values

    def test_process_data(self, downloader_instance, sample_downloaded_df):
        """Tests the data transformation logic."""
        official_df, preliminary_df = downloader_instance.process_data(sample_downloaded_df)
        
        # Check official data
        assert len(official_df) == 1
        assert official_df.iloc[0]["location"] == "CA"
        assert "Region 1" not in official_df["jurisdiction"].values
        assert "date" in official_df.columns
        
        # Check preliminary data
        assert len(preliminary_df) == 1
        assert preliminary_df.iloc[0]["location"] == "TX"

    def test_save_data(self, downloader_instance, tmp_path, monkeypatch):
        """Tests that data is saved correctly to JSON files in multiple locations."""
        official_df = pd.DataFrame([{"location": "CA", "date": pd.Timestamp("2023-01-07"), "totalconfrsvnewadm": 100.0}])
        preliminary_df = pd.DataFrame([{"location": "CA", "date": pd.Timestamp("2023-01-14"), "totalconfrsvnewadm": 110.0}])

        # --- FIX: Change the current directory to tmp_path for this test ---
        monkeypatch.chdir(tmp_path)
        
        downloader_instance.save_data(official_df, preliminary_df)

        # Check that files were created in both target directories
        # The first path is relative to downloader_instance.output_path (which is tmp_path)
        target_dir = downloader_instance.output_path / "nhsn"
        # The second path is now relative to the new current working directory (also tmp_path)
        app_target_dir = Path("app/public/processed_data/nhsn")
        
        ca_payload_path1 = target_dir / "CA_nhsn.json"
        ca_payload_path2 = app_target_dir / "CA_nhsn.json"
        
        assert ca_payload_path1.exists()
        assert ca_payload_path2.exists() # This will now pass

        # Check content of one of the files
        with open(ca_payload_path1) as f:
            data = json.load(f)

        assert data["metadata"]["abbreviation"] == "CA"
        assert data["ground_truth"]["dates"] == ["2023-01-07"]
        assert data["data"]["official"]["totalconfrsvnewadm"] == [100.0]
        assert data["data"]["preliminary"]["totalconfrsvnewadm"] == [110.0]