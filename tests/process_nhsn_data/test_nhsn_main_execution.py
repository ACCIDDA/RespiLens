import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / 'scripts'))

from scripts.process_nhsn_data import main as nhsn_main


@pytest.fixture
def mock_downloader(monkeypatch):
    """Mocks the entire NHSNDataDownloader class and spies on its methods."""
    
    class MockDownloader:
        _init_called = False
        _download_called = False
        _process_called = False
        _save_called = False
        
        def __init__(self, output_path, locations_path):
            MockDownloader._init_called = True
            MockDownloader.init_args = {'output_path': output_path, 'locations_path': locations_path}

        def download_data(self):
            MockDownloader._download_called = True
            return "dummy_raw_dataframe"

        def process_data(self, df):
            assert df == "dummy_raw_dataframe" 
            MockDownloader._process_called = True
            return "dummy_official_df", "dummy_preliminary_df"

        def save_data(self, official_df, preliminary_df):
            assert official_df == "dummy_official_df"
            assert preliminary_df == "dummy_preliminary_df"
            MockDownloader._save_called = True
            
        @classmethod
        def reset(cls):
            cls._init_called = False
            cls._download_called = False
            cls._process_called = False
            cls._save_called = False
            cls.init_args = {}

    monkeypatch.setattr("scripts.process_nhsn_data.NHSNDataDownloader", MockDownloader)
    
    yield MockDownloader
    MockDownloader.reset()


def test_main_happy_path(monkeypatch, tmp_path, mock_downloader):
    """Tests the main execution flow with all arguments."""
    output_dir = tmp_path / "output"
    locations_file = tmp_path / "locations.csv"
    
    test_args = [
        "process_nhsn_data.py",
        "--output-path", str(output_dir),
        "--locations-path", str(locations_file)
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    nhsn_main()

    assert mock_downloader._init_called is True
    assert mock_downloader.init_args['output_path'] == str(output_dir)
    assert mock_downloader.init_args['locations_path'] == str(locations_file)
    
    assert mock_downloader._download_called is True
    assert mock_downloader._process_called is True
    assert mock_downloader._save_called is True

def test_main_exception_handling(monkeypatch, tmp_path, mock_downloader, caplog):
    """Tests that main logs and raises an exception on failure."""
    
    def mock_download_fails(self):
        raise ConnectionError("API is down")
    
    monkeypatch.setattr(mock_downloader, "download_data", mock_download_fails)
    
    test_args = [
        "process_nhsn_data.py",
        "--output-path", str(tmp_path),
        "--locations-path", str(tmp_path / "loc.csv")
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    
    with pytest.raises(ConnectionError):
        nhsn_main()
    
    assert "Failed to download and process NHSN data" in caplog.text
    assert "API is down" in caplog.text