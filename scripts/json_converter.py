"""
A command-line utility to convert external CSV data into RespiLens JSON format.

This script processes CSV files, validates them against required schemas using
a location metadata file, and saves the output as per-location JSON files.

Classes:
    ExternalData: Handles the loading, validation, and conversion of a single
                  source data file into the RespiLens format.

Usage:
    python your_script_name.py --data-path <path/to/data> \\
                               --location-metadata-path <path/to/metadata.json> \\
                               --dataset "MyDataset" \\
                               --output-path <path/to/output>

"""

import argparse
import copy
import json
import logging
import sys
import time  
from pathlib import Path

import pandas as pd
from datetime import date
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from metadata_builder import metadata_builder
from save_respilens_data import save_data, save_metadata

SCRIPT_DIR = Path(__file__).parent
SCHEMA_DIR = SCRIPT_DIR / "schemas"
with open(SCHEMA_DIR / "location-metadata.schema.json", "r") as f:
    LOCATION_METADATA_SCHEMA = json.load(f)
with open(SCHEMA_DIR / "respilens-data.schema.json", "r") as f:
    RESPILENS_DATA_SCHEMA = json.load(f)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LoggedOperation:
    """
    A context manager to log the start, success/failure, and duration of an operation.
    """
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logger.info(f"Starting: {self.name}...")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            logger.info(f"Success: '{self.name}' completed in {duration:.2f}s.")
        else:
            logger.error(f"Failed: '{self.name}' failed after {duration:.2f}s. Reason: {exc_val}")


class ExternalData:
    """
    Loads, validates, and converts external .csv or .json data into a RespiLens-compatible format.

    Attributes:
        dataset: The source your data was pulled fro (e.g., CDC)
        json_struct: The RespiLens JSON structure template, to be recursively populated.
        location_metadata: Location metadata (JSON)
        ext: The file extension of input data
        data: The external data to be converted
        location_col: The name of the column that holds location info
        date_col: The  name of the column that holds date info
        RespiLens_data: Correctly formatted RespiLens JSON data
    """

    def __init__(self, data_path: str | Path, location_metadata_path: str | Path, dataset: str):
        """
        Initialize the ExternalData class.

        Args:
            data_path: Path to the external data to be converted to RespiLens format.
            location_metadata_path: JSON-style metadata for your locations. 
            dataset: What dataset the data is pulled from; e.g., 'CDC'
        """
        logger.info(f"Processing file at {data_path}.")
        self.dataset = dataset
        self.json_struct = {
            "metadata": {
                "dataset": f"{self.dataset}",
                "location": "",
                "series_type": "official"
            },
            "series": {
                "dates": [],
                "columns": {}
            }
        }

        data_path = Path(data_path)
        self.ext = data_path.suffix.lower()

        if self.ext == '.csv':
            with LoggedOperation(f"Reading data from {data_path.name}"):
                self.data = pd.read_csv(data_path)

            self._validate_df()
            self._convert_to_ISO8601_date()

            with LoggedOperation("Loading location metadata"):
                location_metadata_path = Path(location_metadata_path)
                self.location_metadata = json.loads(location_metadata_path.read_text())

            self._validate_and_map_locations()

            self.RespiLens_data = self._convert_from_df()

            with LoggedOperation("Final validation of RespiLens JSON data against schema"):
                for location_entry in self.RespiLens_data.keys():
                    try:
                        validate(instance=self.RespiLens_data[location_entry], schema=RESPILENS_DATA_SCHEMA)
                    except ValidationError as e:
                        raise ValueError(f"Final JSON for location '{location_entry}' failed schema validation.") from e
        else:
            raise ValueError(f"Unsupported file type: {self.ext}. File must be a CSV.")

    def _validate_df(self) -> None:
        """
        Ensure pd.DataFrame to be converted meets requirements:
            - 'location' and 'date' columns are present
            - If locations are listed as FIPS codes, single digits are padded with a leading 0
        """
        with LoggedOperation("DataFrame column and format validation"):
            required_cols = {'location', 'date'}
            missing_cols = required_cols.difference(self.data.columns)
            if missing_cols:
                raise KeyError(f"Input data is missing required columns: {sorted(list(missing_cols))}. Columns are case-sensitive.")

            self.location_col = 'location'
            self.date_col = 'date'

            # Pad FIPS codes with a leading 0 if they are only 1 digit
            self.data[self.location_col] = self.data[self.location_col].astype(str).str.zfill(2)

    def _validate_and_map_locations(self) -> None:
        """
        From location metadata, maps location name, FIPS code, location abbreviation to location abbreviation.

        Ensure that locations provided in DataFrame location column can be mapped (via location metadata) to
        a two-character abbreviation that RespiLens stipulates.
        """
        with LoggedOperation("Location metadata validation and identifier mapping"):
            # Validate the location_metadata structure with jsonschema
            try:
                validate(instance=self.location_metadata, schema=LOCATION_METADATA_SCHEMA)
            except ValidationError as e:
                raise ValueError("Location metadata file does not comply with RespiLens schema.") from e

            # Create a mapping from all possible identifiers to the abbreviation
            loc_identifier_to_abbrv_map = {}
            for entry in self.location_metadata:
                abbreviation = entry['abbreviation']
                loc_identifier_to_abbrv_map[entry['name'].lower()] = abbreviation
                loc_identifier_to_abbrv_map[entry['location']] = abbreviation
                loc_identifier_to_abbrv_map[entry['abbreviation'].lower()] = abbreviation

            # Use the map to match abbreviations for locations in the data
            self.data['abbreviation'] = self.data[self.location_col].astype(str).str.lower().map(loc_identifier_to_abbrv_map)

            if self.data['abbreviation'].isna().any():
                locs_with_nans = self.data[self.data['abbreviation'].isna()][self.location_col].unique()
                raise ValueError(f"Could not find a metadata abbreviation match for the following locations: {list(locs_with_nans)}")

    def _convert_to_ISO8601_date(self) -> None:
        """
        Converts self.data dates to strs in ISO 8601 format.
        """
        with LoggedOperation("Date column conversion to ISO 8601 format"):
            try:
                self.data[self.date_col] = pd.to_datetime(
                    self.data[self.date_col],
                    errors='raise',
                    format='mixed'
                ).dt.strftime('%Y-%m-%d')
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to normalize dates in column '{self.date_col}'. Ensure all dates are in a recognizable format.") from e

    def _convert_from_df(self) -> dict:
        """
        Converts external .csv data into RespiLens-formatted JSON data.

        Returns:
            A dict where each entry is a unique location from .CSV.
        """
        with LoggedOperation("Data conversion to RespiLens JSON format"):
            return_json = {}
            unique_locations = self.data['abbreviation'].unique()

            for location_abbrv in unique_locations:
                json_entry = copy.deepcopy(self.json_struct)
                json_entry["metadata"]["location"] = location_abbrv

                current_loc_df = self.data[self.data['abbreviation'] == location_abbrv]
                json_entry["series"]["dates"] = list(current_loc_df[self.date_col])

                for column in current_loc_df.columns:
                    if column not in (self.location_col, self.date_col, 'abbreviation'):
                        json_entry["series"]["columns"][column] = list(current_loc_df[column])

                return_json[location_abbrv] = json_entry

        return return_json


def main():
    """
    Main execution function.
    """

    parser = argparse.ArgumentParser(description="Convert external data to RespiLens JSON format.")
    parser.add_argument("--data-path",
                        type=str,
                        required=True,
                        help="Path to data, either a single file or directory with multiple files. File(s) must be CSVs.")
    parser.add_argument("--location-metadata-path",
                        type=str,
                        required=True,
                        help="Path to location metadata (must be a single JSON file)")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="What dataset the data is pulled from (for metadata purposes).")
    parser.add_argument("--output-path",
                        type=str,
                        required=True,
                        help="Path to directory to save data to.")
    args = parser.parse_args()
    
    with LoggedOperation(f"Entire conversion process for dataset '{args.dataset}'"):
        if not (Path(args.location_metadata_path).is_file()) or (Path(args.location_metadata_path).suffix.lower() != '.json'):
            logger.error(f"The provided location metadata path '{args.location_metadata_path}' is not a valid JSON file.")
            sys.exit(1)

        list_of_converted_RespiLens_data_objects = []
        unconverted_file_paths = []
        
        with LoggedOperation("Identifying and processing all source files"):
            files_to_process = []
            if Path(args.data_path).is_file():
                files_to_process.append(Path(args.data_path))
            elif Path(args.data_path).is_dir():
                files_to_process = [f for f in Path(args.data_path).glob('*.csv') if f.is_file()]
            
            logger.info(f"Found {len(files_to_process)} .csv file(s) to process.")

            for file in files_to_process:
                try:
                    converter_object = ExternalData(file, args.location_metadata_path, args.dataset)
                    list_of_converted_RespiLens_data_objects.append((converter_object, file))
                except Exception as e:
                    logger.error(f"Failed to process file: {file.name}. Skipping. Reason: {e}")
                    unconverted_file_paths.append(file)
        
        if unconverted_file_paths and not list_of_converted_RespiLens_data_objects:
            raise RuntimeError("Failed to convert all files provided. Check logs for details.")

        with LoggedOperation("Building and saving new metadata"):
            metadata = metadata_builder(args.dataset)
            logger.info("Missing metadata fields can be filled in manually after completion.")
            save_metadata(metadata, args.output_path)

        with LoggedOperation("Saving all converted RespiLens data"):
            unsaved_files = []
            saved_count = 0
            for converter_object, source_file in list_of_converted_RespiLens_data_objects:
                for location, location_data in converter_object.RespiLens_data.items():
                    try:
                        save_data(location_data, args.output_path)
                        saved_count += 1
                    except FileExistsError as e:
                        logger.critical(f"FATAL: Attempted to overwrite file for location '{location}'.")
                        logger.critical(f"If attempting to update data, please ensure directory is clear first.")
                        raise FileExistsError(f"{e}")
                    except Exception as e:
                        unsaved_files.append((location, source_file.name, e))

            logger.info(f"Successfully saved data for {saved_count} locations.")
            if unsaved_files:
                logger.warning(f"Failed to save data for {len(unsaved_files)} locations:")
                for loc, fname, err in unsaved_files:
                    logger.warning(f"  - Location: {loc} (from file {fname}). Reason: {err}")

if __name__ == "__main__":
    main()