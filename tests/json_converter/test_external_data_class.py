import pytest
import pandas as pd
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / 'scripts'))

from scripts.json_converter import ExternalData


@pytest.fixture
def valid_location_metadata(tmp_path):
    """
    Creates a valid location_metadata.json file in a temporary directory.
    """
    metadata = [
        {
            "name": "Alabama",
            "abbreviation": "AL",
            "location": "01",
            "population": 5024279
        },
        {
            "name": "California",
            "abbreviation": "CA",
            "location": "06",
            "population": 39538223
        }
    ]
    metadata_path = tmp_path / "location_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    return metadata_path


@pytest.fixture
def invalid_location_metadata(tmp_path):
    """
    Creates an invalid location_metadata.json file (missing 'abbreviation').
    """
    metadata = [
        {
            "name": "Alabama",
            "location": "01",
            "population": 5024279
        }
    ]
    metadata_path = tmp_path / "invalid_location_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    return metadata_path


@pytest.fixture
def valid_csv_data(tmp_path):
    """
    Creates a valid CSV data file in a temporary directory.
    """
    data = {
        "date": ["2023-01-01", "2023-01-02", "2023-01-01"],
        "location": ["01", "01", "California"], # Note: '1' will be tested for zfill
        "cases": [100, 110, 200],
        "deaths": [5, 6, 10]
    }
    df = pd.DataFrame(data)
    data_path = tmp_path / "valid_data.csv"
    df.to_csv(data_path, index=False)
    return data_path


@pytest.fixture
def csv_with_unmappable_location(tmp_path):
    """
    Creates a CSV with a location that is not in the valid metadata.
    """
    data = {
        "date": ["2023-01-01"],
        "location": ["Texas"], # Texas is not in the sample metadata
        "cases": [300]
    }
    df = pd.DataFrame(data)
    data_path = tmp_path / "unmappable_data.csv"
    df.to_csv(data_path, index=False)
    return data_path


@pytest.fixture
def csv_with_missing_column(tmp_path):
    """
    Creates a CSV that is missing the required 'date' column.
    """
    data = {
        "location": ["01"],
        "cases": [100]
    }
    df = pd.DataFrame(data)
    data_path = tmp_path / "missing_col_data.csv"
    df.to_csv(data_path, index=False)
    return data_path

@pytest.fixture
def csv_with_bad_date(tmp_path):
    """
    Creates a CSV with a date that cannot be parsed.
    """
    data = {
        "date": ["not-a-real-date"],
        "location": ["01"],
        "cases": [100]
    }
    df = pd.DataFrame(data)
    data_path = tmp_path / "bad_date_data.csv"
    df.to_csv(data_path, index=False)
    return data_path


class TestExternalData:
    """
    A test class to group all tests for the ExternalData class.
    """

    def test_init_success(self, valid_csv_data, valid_location_metadata):
        """
        Tests successful instantiation of ExternalData with valid inputs.
        """
        converter = ExternalData(
            data_path=valid_csv_data,
            location_metadata_path=valid_location_metadata,
            dataset="TestDataset"
        )
        assert converter.RespiLens_data is not None
        assert "AL" in converter.RespiLens_data
        assert "CA" in converter.RespiLens_data
        assert len(converter.RespiLens_data) == 2
        alabama_data = converter.RespiLens_data["AL"]
        assert alabama_data["metadata"]["dataset"] == "TestDataset"
        assert alabama_data["metadata"]["location"] == "AL"
        assert sorted(alabama_data["series"]["dates"]) == ["2023-01-01", "2023-01-02"]
        assert "cases" in alabama_data["series"]["columns"]
        assert "deaths" in alabama_data["series"]["columns"]
        assert alabama_data["series"]["columns"]["cases"] == [100, 110]


    def test_init_unsupported_file_type(self, tmp_path, valid_location_metadata):
        """
        Tests that initializing with an unsupported file type raises a ValueError.
        """
        unsupported_file = tmp_path / "data.txt"
        unsupported_file.touch()
        with pytest.raises(ValueError, match="Unsupported file type: .txt"):
            ExternalData(unsupported_file, valid_location_metadata, "TestDataset")


    def test_validate_df_missing_required_column(self, csv_with_missing_column, valid_location_metadata):
        """
        Tests that _validate_df correctly raises a KeyError for missing columns.
        """
        with pytest.raises(KeyError, match="Input data is missing required columns: \\['date'\\]"):
            ExternalData(csv_with_missing_column, valid_location_metadata, "TestDataset")


    def test_validate_and_map_locations_invalid_metadata_schema(self, valid_csv_data, invalid_location_metadata):
        """
        Tests that a validation error is raised if the location metadata is invalid.
        """
        with pytest.raises(ValueError, match="Location metadata does not comply with RespiLens schema."):
            ExternalData(valid_csv_data, invalid_location_metadata, "TestDataset")


    def test_validate_and_map_locations_unmappable_location(self, csv_with_unmappable_location, valid_location_metadata):
        """
        Tests that a ValueError is raised for locations in data not found in metadata.
        """
        with pytest.raises(ValueError, match="Could not find a metadata abbreviation match for the following locations: \\['Texas'\\]"):
            ExternalData(csv_with_unmappable_location, valid_location_metadata, "TestDataset")


    def test_convert_to_iso8601_date_failure(self, csv_with_bad_date, valid_location_metadata):
        """
        Tests that date conversion fails for un-parseable date strings.
        """
        with pytest.raises(ValueError, match="Failed to normalize dates in column 'date'"):
            ExternalData(csv_with_bad_date, valid_location_metadata, "TestDataset")
            
    def test_fips_code_padding(self, tmp_path, valid_location_metadata):
        """
        Tests that single-digit FIPS codes are correctly padded with a leading zero.
        """
        data = {
            "date": ["2023-01-01"],
            "location": ["6"], # California's FIPS is 06, testing padding '6' -> '06'
            "cases": [200]
        }
        df = pd.DataFrame(data)
        data_path = tmp_path / "fips_padding_test.csv"
        df.to_csv(data_path, index=False)
        converter = ExternalData(data_path, valid_location_metadata, "TestDataset")
        assert "CA" in converter.RespiLens_data
        assert converter.RespiLens_data["CA"]["metadata"]["location"] == "CA"
