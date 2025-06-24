import pytest
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / 'scripts'))

from scripts.save_respilens_data import save_data, save_metadata

@pytest.fixture
def valid_respilens_data():
    """
    Provides a dictionary that is valid against the respilens-data.schema.json.
    """
    return {
        "metadata": {
            "dataset": "TestSet",
            "location": "CA",
            "series_type": "official"
        },
        "series": {
            "dates": ["2023-01-01", "2023-01-02"],
            "columns": {
                "cases": [100, 110],
                "deaths": [5, 6]
            }
        }
    }

@pytest.fixture
def invalid_respilens_data():
    """
    Provides a dictionary that is invalid (missing the required 'series' key).
    """
    return {
        "metadata": {
            "dataset": "TestSet",
            "location": "NY",
            "series_type": "official"
        }
        # "series" key is missing
    }

@pytest.fixture
def valid_metadata():
    """
    Provides a simple dictionary to act as valid metadata.
    """
    return {
        "dataset": "TestSet",
        "author": "Test Author",
        "source_files": ["file1.csv", "file2.csv"]
    }


class TestSaveData:
    """A test class for the save_data function."""

    def test_save_data_success(self, valid_respilens_data, tmp_path):
        """
        Tests that save_data correctly saves a valid data dictionary
        to a file with the correct name and content.
        """
        output_path = tmp_path
        save_data(valid_respilens_data, output_path)
        expected_file = output_path / "CA.json"
        assert expected_file.exists()
        with open(expected_file, "r") as f:
            saved_data = json.load(f)
        assert saved_data == valid_respilens_data

    def test_save_data_creates_directory(self, valid_respilens_data, tmp_path):
        """
        Tests that save_data creates the output directory if it doesn't exist.
        """
        non_existent_path = tmp_path / "new_output_dir"
        assert not non_existent_path.exists()
        save_data(valid_respilens_data, non_existent_path)
        assert non_existent_path.exists()
        assert (non_existent_path / "CA.json").exists()

    def test_save_data_invalid_schema(self, invalid_respilens_data, tmp_path):
        """
        Tests that save_data raises a ValueError if the provided data
        does not conform to the RespiLens data schema.
        """
        with pytest.raises(ValueError, match="Data does not match RespiLens jsonschema."):
            save_data(invalid_respilens_data, tmp_path)


class TestSaveMetadata:
    """A test class for the save_metadata function."""

    def test_save_metadata_success(self, valid_metadata, tmp_path):
        """
        Tests that save_metadata correctly saves a valid metadata dictionary
        to 'metadata.json'.
        """
        output_path = tmp_path
        save_metadata(valid_metadata, output_path)
        expected_file = output_path / "metadata.json"
        assert expected_file.exists()
        with open(expected_file, "r") as f:
            saved_metadata = json.load(f)
        assert saved_metadata == valid_metadata

    def test_save_metadata_creates_directory(self, valid_metadata, tmp_path):
        """
        Tests that save_metadata creates the output directory if it doesn't exist.
        """
        non_existent_path = tmp_path / "new_metadata_dir"
        assert not non_existent_path.exists()
        save_metadata(valid_metadata, non_existent_path)
        assert non_existent_path.exists()
        assert (non_existent_path / "metadata.json").exists()
