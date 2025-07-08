import pytest
from pathlib import Path
import sys
import logging

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / 'scripts'))

from json_converter import main as json_converter_main


@pytest.fixture
def mock_dependencies(monkeypatch):
    """
    A fixture to mock all external dependencies of the main() function.
    """
    class MockExternalData:
        def __init__(self, data_path, location_metadata_path, dataset):
            if 'bad' in str(data_path):
                 raise ValueError("Simulated conversion failure")
            if Path(data_path).suffix.lower() != '.csv':
                raise ValueError("Unsupported file type")

            self.data_path = data_path
            self.dataset = dataset
            self.RespiLens_data = {"NY": {"metadata": {"location": "NY"}, "series": {}}}

        def __repr__(self):
            return f"MockExternalData({self.data_path})"

    mock_builder = lambda dataset: {"mock": "metadata"}
    mock_save_metadata = lambda metadata, path: None
    mock_save_data = lambda data, path: None

    monkeypatch.setattr("json_converter.ExternalData", MockExternalData)
    monkeypatch.setattr("json_converter.metadata_builder", mock_builder)
    monkeypatch.setattr("json_converter.save_metadata", mock_save_metadata)
    monkeypatch.setattr("json_converter.save_data", mock_save_data)


def test_main_successful_run(monkeypatch, tmp_path, mock_dependencies):
    """
    Tests the main function's happy path with a single valid file.
    """
    data_file = tmp_path / "data.csv"
    data_file.touch()
    metadata_file = tmp_path / "metadata.json"
    metadata_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    test_args = [
        "json_converter.py",
        "--data-path", str(data_file),
        "--location-metadata-path", str(metadata_file),
        "--dataset", "MyTest",
        "--output-path", str(output_dir)
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    json_converter_main()


def test_main_directory_run(monkeypatch, tmp_path): 
    """
    Tests the main function's happy path when given a directory of files.
    """
    call_log = []
    class LoggingMockExternalData:
        def __init__(self, data_path, location_metadata_path, dataset):
            # This mock will fail for non-csv files but log successes
            if Path(data_path).suffix.lower() != '.csv':
                raise ValueError("Unsupported file type for this mock")
            
            # If init is successful, log the call
            call_log.append(data_path)
            
            # Add necessary attributes for the rest of the script to run
            self.data_path = data_path
            self.dataset = dataset
            self.RespiLens_data = {"NY": {"metadata": {"location": "NY"}, "series": {}}}

    # Patch all the necessary dependencies for this test
    monkeypatch.setattr("json_converter.ExternalData", LoggingMockExternalData)
    monkeypatch.setattr("json_converter.metadata_builder", lambda dataset: {"mock": "metadata"})
    monkeypatch.setattr("json_converter.save_metadata", lambda metadata, path: None)
    monkeypatch.setattr("json_converter.save_data", lambda data, path: None)
 

    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "data1.csv").touch()
    (data_dir / "data2.csv").touch()
    (data_dir / "other.txt").touch() 
    
    metadata_file = tmp_path / "metadata.json"
    metadata_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    test_args = [
        "json_converter.py",
        "--data-path", str(data_dir),
        "--location-metadata-path", str(metadata_file),
        "--dataset", "MyTest",
        "--output-path", str(output_dir)
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    
    json_converter_main() 

    # Assert that ExternalData was successfully initialized only for the CSV files
    assert len(call_log) == 2
    assert data_dir / "data1.csv" in call_log
    assert data_dir / "data2.csv" in call_log


def test_main_invalid_metadata_path(monkeypatch, tmp_path):
    """
    Tests that the script fails if the location metadata path is invalid.
    """
    test_args = [
        "json_converter.py",
        "--data-path", "dummy",
        "--location-metadata-path", str(tmp_path),
        "--dataset", "MyTest",
        "--output-path", "dummy"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as e:
        json_converter_main()
    assert e.type == SystemExit
    assert e.value.code == 1


def test_main_partial_conversion_failure(monkeypatch, tmp_path, mock_dependencies, caplog):
    """
    Tests that main logs failures but continues if some files fail conversion.
    """
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "good_data.csv").touch()
    (data_dir / "bad_data.csv").touch()

    metadata_file = tmp_path / "metadata.json"
    metadata_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    test_args = [
        "json_converter.py",
        "--data-path", str(data_dir),
        "--location-metadata-path", str(metadata_file),
        "--dataset", "MyTest",
        "--output-path", str(output_dir)
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with caplog.at_level(logging.INFO):
        json_converter_main()

    assert "Failed to convert 1/1 files:" in caplog.text
    assert "bad_data.csv" in caplog.text
    assert "Proceeding for successfully converted files." in caplog.text
    assert "Saving data..." in caplog.text


def test_main_total_conversion_failure(monkeypatch, tmp_path, mock_dependencies):
    """
    Tests that main raises a RuntimeError if ALL files fail conversion.
    """
    monkeypatch.setattr("json_converter.ExternalData", lambda a, b, c: (_ for _ in ()).throw(ValueError("Always fail")))

    data_file = tmp_path / "any_file.csv"
    data_file.touch()
    metadata_file = tmp_path / "metadata.json"
    metadata_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    test_args = [
        "json_converter.py",
        "--data-path", str(data_file),
        "--location-metadata-path", str(metadata_file),
        "--dataset", "MyTest",
        "--output-path", str(output_dir)
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(RuntimeError, match="Failed to convert all files provided."):
        json_converter_main()


def test_main_data_saving_failure(monkeypatch, tmp_path, mock_dependencies, caplog):
    """
    Tests that main logs an error if save_data fails.
    """
    def mock_save_data_fails(data, path):
        raise IOError("Disk is full")

    monkeypatch.setattr("json_converter.save_data", mock_save_data_fails)

    data_file = tmp_path / "good_data.csv"
    data_file.touch()
    metadata_file = tmp_path / "metadata.json"
    metadata_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    test_args = [
        "json_converter.py",
        "--data-path", str(data_file),
        "--location-metadata-path", str(metadata_file),
        "--dataset", "MyTest",
        "--output-path", str(output_dir)
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with caplog.at_level(logging.INFO):
        json_converter_main()

    assert "Failed to save 1 files:" in caplog.text
    assert f"NY.json from your file {data_file}" in caplog.text