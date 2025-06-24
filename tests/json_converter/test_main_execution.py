import pytest
from pathlib import Path
import sys

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
            if Path(data_path).suffix.lower() != '.csv':
                raise ValueError("Unsupported file type")

            self.data_path = data_path
            self.dataset = dataset
            self.RespiLens_data = {"NY": {"metadata": {}, "series": {}}}

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


def test_main_directory_run(monkeypatch, tmp_path, mock_dependencies):
    """
    Tests the main function's happy path when given a directory of files.
    """
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "data1.csv").touch()
    (data_dir / "data2.csv").touch()
    (data_dir / "other.txt").touch() # This file should be ignored
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
    call_log = []
    original_mock = sys.modules["json_converter"].ExternalData

    def logging_mock_init(self, data_path, location_metadata_path, dataset):
        original_mock.__init__(self, data_path, location_metadata_path, dataset)
        call_log.append(data_path)

    monkeypatch.setattr("json_converter.ExternalData", type('LoggingMock', (original_mock,), {'__init__': logging_mock_init}))
    json_converter_main()
    assert len(call_log) == 2
    assert data_dir / "data1.csv" in call_log
    assert data_dir / "data2.csv" in call_log


def test_main_invalid_metadata__path(monkeypatch, tmp_path):
    """
    Tests that the script fails if the location metadata path is invalid.
    """
    test_args = [
        "json_converter.py",
        "--data-path", "dummy",
        "--location-metadata-path", str(tmp_path), # tmp_path is a directory; main() needs a single file
        "--dataset", "MyTest",
        "--output-path", "dummy"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as e:
        json_converter_main()
    assert e.type == SystemExit
    assert e.value.code == 1
