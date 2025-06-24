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
from pathlib import Path

import pandas as pd
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from metadata_builder import metadata_builder
from save_respilens_data import save_data, save_metadata

SCRIPT_DIR = Path(__file__).parent
with open(SCRIPT_DIR / "location-metadata.schema.json", "r") as f:
    LOCATION_METADATA_SCHEMA = json.load(f)
with open(SCRIPT_DIR / "respilens-data.schema.json", "r") as f:
    RESPILENS_DATA_SCHEMA = json.load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            self.data = pd.read_csv(data_path)

            # Validate DF contents
            self._validate_df()
            self._convert_to_ISO8601_date()

            # Validate location metadata using jsonschema, create map from all possible identifiers to abbreviation
            location_metadata_path = Path(location_metadata_path)
            self.location_metadata = json.loads(location_metadata_path.read_text())
            self._validate_and_map_locations()

            # Convert validated DataFrame to RespiLens-formatted JSON, confirm it is valid RespiLens data (final validation step)
            self.RespiLens_data = self._convert_from_df()
            logger.info("Validating final RespiLens JSON data...")
            for location_entry in self.RespiLens_data.keys():
                try:
                    validate(instance=self.RespiLens_data[location_entry], schema=RESPILENS_DATA_SCHEMA)
                except ValidationError as e:
                    raise ValueError(f"Final RespiLens JSON did not match jsonschema. First failed location entry is {location_entry}.") from e
            logger.info("Success.")

        else:
            raise ValueError(f"Unsupported file type: {self.ext}. File must be a CSV.")


    def _validate_df(self) -> None:
        """
        Ensure pd.DataFrame to be converted meets requirements:
            - 'location' and 'date' columns are present
            - If locations are listed as FIPS codes, single digits are padded with a leading 0
        """

        logger.info("Ensuring that DataFrame data contains a location and date column")
        # Ensure that data contains columns 'location' and 'date', stop execution if not
        # Define the exact required columns as a set
        required_cols = {'location', 'date'}
        missing_cols = required_cols.difference(self.data.columns)
        if missing_cols:
            raise KeyError(f"Input data is missing required columns: {sorted(list(missing_cols))}. Columns are case-sensitive.")
        self.location_col = 'location'
        self.date_col = 'date'

        # Pad FIPS codes with a leading 0 if they are only 1 digit
        self.data[self.location_col] = self.data[self.location_col].astype(str).str.zfill(2)

        logger.info("Success.")


    def _validate_and_map_locations(self) -> None:
        """
        From location metadata, maps location name, FIPS code, location abbreviation to location abbreviation.

        Ensure that locations provided in DataFrame location column can be mapped (via location metadata) to
        a two-character abbreviation that RespiLens stipulates.
        """

        # Validate location_metadata structure using .validate_jsonschema() method
        logger.info("Validating location metadata...")
        try:
            validate(instance=self.location_metadata, schema=LOCATION_METADATA_SCHEMA)
        except ValidationError as e:
            logger.info("Cannot proceed with data conversion if location metadata does not match RespiLens standards.")
            raise ValueError("Location metadata does not comply with RespiLens schema.") from e
        logger.info("Success.")

        # Map and match metadata locations to DF locations
        logger.info("Mapping various location identifiers to location abbreviations...")
        loc_identifier_to_abbrv_map = {}
        for entry in self.location_metadata:
            abbreviation = entry['abbreviation']
            loc_identifier_to_abbrv_map[entry['name'].lower()] = abbreviation
            loc_identifier_to_abbrv_map[entry['location']] = abbreviation
            loc_identifier_to_abbrv_map[entry['abbreviation'].lower()] = abbreviation
        logger.info("Success")
        logger.info(f"Matching locations in '{self.location_col}' col to metadata...")
        loc_abbrv_list = self.data[self.location_col].astype(str).str.lower().map(loc_identifier_to_abbrv_map).tolist()
        self.data['abbreviation'] = loc_abbrv_list
        if pd.isna(self.data['abbreviation']).any():
            locs_with_nans = self.data[pd.isna(self.data['abbreviation'])][self.location_col].unique()
            raise ValueError(f"Could not find a metadata abbreviation match for the following locations: {list(locs_with_nans)}")
        logger.info("Success.")


    def _convert_to_ISO8601_date(self) -> None: 
        """
        Converts self.data dates to strs in ISO 8601 format.
        """

        logger.info("Converting dates into ISO 8601 strings...")
        try:
            self.data[self.date_col] = pd.to_datetime(
                self.data[self.date_col], 
                errors='raise', 
                format='mixed'
            ).dt.strftime('%Y-%m-%d')
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to normalize dates in column '{self.date_col}'. Ensure all dates are in recognizable format.") from e
        logger.info("Success.")
        

    def _convert_from_df(self) -> dict: 
        """
        Converts external .csv data into RespiLens-formatted JSON data.

        Returns:
            A dict where each unique location has a RespiLens JSON entry.
        """

        # Create JSON to store the properly formatted data
        return_json = {}

        logger.info("Converting data to RespiLens JSON format...")
        # Populate a JSON entry with properly formatted unique location data
        unique_locations = set(list(self.data['abbreviation'])) 
        for location_abbrv in unique_locations:
            json_entry = copy.deepcopy(self.json_struct)
            json_entry["metadata"]["location"] = location_abbrv # add loc to metadata header
            current_loc_df = self.data[self.data['abbreviation'] == location_abbrv]
            json_entry["series"]["dates"] = list(current_loc_df[self.date_col]) # add date range to dates key
            for column in current_loc_df.columns:
                if column in (self.location_col, self.date_col, 'abbreviation'):
                    continue
                else:
                    json_entry["series"]["columns"][column] = list(current_loc_df[column])
            
            # Add to larger JSON
            return_json[location_abbrv] = json_entry
        
        logger.info("Success.")
        return return_json


def main(): 
    """
    Main execution function.
    """

    parser = argparse.ArgumentParser(description = "Convert external data to RespiLens JSON format.")
    parser.add_argument("--data-path",
                        type = str,
                        required = True,
                        help = "Path to data, either a single file or directory with multiple files. File(s) must be CSVs.")
    parser.add_argument("--location-metadata-path",
                        type = str,
                        required = True,
                        help = "Path to location metadata (must be a single JSON file)")
    parser.add_argument("--dataset",
                        type = str,
                        required = True,
                        help = "What dataset the data is pulled from (for metadata purposes).")
    parser.add_argument("--output-path", 
                        type = str,
                        required = True,
                        help = "Path to directory to save data to.")
    args = parser.parse_args()
    if not (Path(args.location_metadata_path).is_file()) or (Path(args.location_metadata_path).suffix.lower() != '.json'):
        logger.error(f"The provided location metadata path '{args.location_metadata_path}' is not a valid JSON file.")
        sys.exit(1)
    
    files_to_process = []
    unconverted_file_paths = []
    converted_file_paths = []
    list_of_converted_RespiLens_data_objects = []

    if Path(args.data_path).is_file():
        files_to_process.append(args.data_path)
    elif Path(args.data_path).is_dir():
        files_to_process = [f for f in Path(args.data_path).glob('*') if f.is_file()] 
    
    for file in files_to_process:
        try:
            converter_object = ExternalData(file, args.location_metadata_path, args.dataset)
            converted_file_paths.append(file)
            list_of_converted_RespiLens_data_objects.append((converter_object, file))
        except Exception:
            unconverted_file_paths.append(file)
    if unconverted_file_paths:
        if not converted_file_paths: 
            raise RuntimeError("Failed to convert all files provided. Ensure all data is a CSV and contains 'location' and 'date' columns.")
        logger.info(f"Failed to convert {len(unconverted_file_paths)}/{len(converted_file_paths)} files:")
        for bad_file in unconverted_file_paths:
            logger.info(f"{bad_file.name}")
        logger.info("Proceeding for successfully converted files.")

    logger.info("Building metadata...")    
    metadata = metadata_builder(args.dataset)
    logger.info("Success, missing metadata fields can be filled in manually.")
    logger.info("Saving metadata...")
    try:
        save_metadata(metadata, args.output_path)
    except Exception as e:
        raise RuntimeError("Failed to save metadata.") from e
    logger.info("Success.")
    
    unsaved_files = []
    saved_files = []
    logger.info("Saving data...") 
    for entry in list_of_converted_RespiLens_data_objects:
        for location, location_data in entry[0].RespiLens_data.items(): 
            try:
                save_data(location_data, args.output_path)
                saved_files.append((location, entry[1]))
            except Exception as e:
                unsaved_files.append((location, entry[1]))
        if unsaved_files:
            logger.info(f"Failed to save {len(unsaved_files)} files:")
            for unsaved_file_tuple in unsaved_files:
                logger.info(f"{unsaved_file_tuple[0]}.json from your file {unsaved_file_tuple[1]}")

    logger.info("Process complete.")
    

if __name__ == "__main__":
    main()