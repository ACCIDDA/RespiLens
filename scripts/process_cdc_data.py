"""
This module facilitates the retrieval, transformation, and storage of data from the CDC's public API.

Data can either be stored as raw json, or as a pd.DataFrame. Metadata is always stored as raw json.

Classes:
    CDCData: Handles downloading, transforming, and saving data from a specific CDC dataset resource.
"""

import argparse
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from jsonschema import validate
from jsonschema.exceptions import ValidationError

import pandas as pd
import requests

from metadata_builder import metadata_builder
from save_RespiLens_data import save_data, save_metadata

SCRIPT_DIR = Path(__file__).parent
SCHEMA_DIR = SCRIPT_DIR / "schemas"
with open(SCHEMA_DIR / "respilens-data.schema.json", "r") as f:
    RESPILENS_DATA_SCHEMA = json.load(f)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CDCData:
    
    def __init__(self, resource_id: str, output_path: str):
        """
        Initialize the CDCData class.
        
        Args:
            resource_id: The unique resource identifier string.
            output_path: Path for data to be stored at, if saved.
        """
        
        self.resource_id = resource_id
        self.data_url = "https://data.cdc.gov/resource/" + f"{resource_id}.json"
        self.metadata_url = "https://data.cdc.gov/api/views/" + f"{resource_id}.json"
        self.output_path = Path(output_path)
    
    def download_cdc_data(self, replace_column_names: bool = True) -> dict: 
        """
        Retrieves and stores data + metadata for specified resource_id.
        
        Arg:
            replace_column_names: Flag to replace short column names with long names.
        
        Returns:
            A dictionary with;
                'data_as_DF' storing data in a pd.DataFrame ('jursidiction' and 'weekendingdates' columns are 
                    standardized as 'location' and 'date'),
                'data' storing data in json format for each location, 
                'CDC_metadata' storing metadata from API as a dictionary of raw json, and
                'respilens_metadata' storing RespiLens-relevant metadata as a dictionary of raw json.
        """
        
        # Initialize dictionary to hold output
        output = {}
        
        # Retrieve metadata from endpoint
        logger.info(f"Retrieving metadata from {self.metadata_url}.")
        metadata_response = requests.get(self.metadata_url)
        CDC_metadata = metadata_response.json() 

        # Build RespiLens-relevant metadata
        # output["respilens_metadata"] = self.build_respilens_metadata()
        output["respilens_metadata"] = metadata_builder(
            shortName="nhsn",
            fullName="National Healthcare Safety Network (NHSN)",
            defaultView="nhsn",
            datasetType="timeseries")

          
        # Retrieve data from endpoint, convert to pd.DataFrame
        logger.info(f"Retrieving data from {self.data_url}.") 
        data_list = self.retrieve_data_from_endpoint_aslist()
        data = pd.DataFrame(data_list)
        data['weekendingdate'] = pd.to_datetime(data['weekendingdate']).dt.strftime('%Y-%m-%d')
        if replace_column_names:
            data = self.replace_column_names(data, CDC_metadata)
            output["data_as_DF"] = data
            unique_regions = set(data["Geographic aggregation"])

            # Coerce into correct json format
            jsons = {}
            for region in unique_regions:
                json_struct = { 
                "metadata": {
                    "dataset": "CDC",
                    "location": f"{region}",
                    "series_type": "official"
                    },
                "series": {
                    "dates": [],
                    "columns": {}
                    }
                }
                current_loc = data[data["Geographic aggregation"] == region]
                json_struct["series"]["dates"] = list(current_loc["Week Ending Date"])
                for column in current_loc.columns:
                    json_struct["series"]["columns"][column] = list(current_loc[column]) 
            
                del json_struct["series"]["columns"]["Geographic aggregation"]
                del json_struct["series"]["columns"]["Week Ending Date"] 
                jsons[region] = json_struct
        else:
            output["data_as_DF"] = data
            unique_regions = set(data["jurisdiction"])

            # Coerce into correct json format
            jsons = {}
            for region in unique_regions:
                json_struct = { 
                "metadata": {
                    "dataset": "CDC",
                    "location": f"{region}",
                    "series_type": "official"
                    },
                "series": {
                    "dates": [],
                    "columns": {}
                    }
                }
                current_loc = data[data["jurisdiction"] == region]
                json_struct["series"]["dates"] = list(current_loc["weekendingdate"])
                for column in current_loc.columns:
                    json_struct["series"]["columns"][column] = list(current_loc[column]) 
            
                del json_struct["series"]["columns"]["jurisdiction"]
                del json_struct["series"]["columns"]["weekendingdate"] 
                jsons[region] = json_struct
        
        # Standardize column names for the output DataFrame to ensure RespiLens compatibility.
        df = output["data_as_DF"]
        rename_map = {
            "Geographic aggregation": "location",
            "jurisdiction": "location",
            "Week Ending Date": "date",
            "weekendingdate": "date"
        }
        # Select only the columns that exist in the DataFrame to avoid KeyErrors
        actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df.rename(columns=actual_rename_map, inplace=True)
        output["data_as_DF"] = df

        # Add data and metadata variations to output dict
        output["CDC_metadata"] = CDC_metadata  
        output["data"] = jsons
        
        logging.info("Success.")
        return output
    
    def retrieve_data_from_endpoint_aslist(self) -> list[dict]:
        """
        Download CDC data from a given endpoint with pagination.
            
        Returns:
            A list of dictionaries containing the json data for specified resource.  
        """
        
        all_data = []
        offset = 0
        batch_size = 1000
        
        # Request data from API in batches
        while True:
            logger.info(f"Downloading records {offset} through {offset + batch_size}")
            params = {
                "$limit": batch_size,
                "$offset": offset
            }
            
            try: 
                data_response = requests.get(self.data_url, params = params)
                data_response.raise_for_status()
                batch_data = data_response.json()
                
                if not batch_data: # No more data
                    break
                    
                all_data.extend(batch_data)
                offset += batch_size
                time.sleep(0.1) # Limit rate to prevent overload
            
            except Exception as e:
                logger.error(f"Error downloading data: {str(e)}")
                break
                
        return all_data
    
    
    def replace_column_names(self, data: pd.DataFrame, CDC_metadata: dict) -> pd.DataFrame:
        """
        Replace short-form column names with long-form column names.
        
        Args:
            data: A pd.DataFrame of the data
            CDC_metadata: A dictionary of metadata containing column name information
            
        Returns:
            A pd.DataFrame with long-form column names.
        """
        
        # Pull long column names from metadata
        long_column_names = []
        for column_index in range(len(CDC_metadata['columns'])):
            long_column_names.append(CDC_metadata['columns'][column_index]['name'])
            
        # Replace columns
        if isinstance(data, pd.DataFrame):
            data.columns = long_column_names
            return data
        
        # Edge case
        else:
            raise TypeError("data input must be pd.DataFrame.")
    
    def save_data_csv(self, data: pd.DataFrame, respilens_metadata: dict) -> None:
        """
        A method to save data at specified output_path as a CSV.
        
        Args:
            data: A pd.DataFrame containing the data.
            respilens_metadata: A dict containing the metadata.
        """
        
        # Create directories
        output_directory = self.output_path 
        os.makedirs(output_directory, exist_ok=True)
        logger.info(f"Saving data as CSV to {output_directory}...")
        
        # Save data to CSVs, metadata to json
        unique_regions = set(data["location"])
        for region in unique_regions:
            current_loc = data[data["location"] == region]
            current_loc.to_csv(f"{output_directory}/{region}.csv", index = False)
        with open(f"{output_directory}/metadata.json", "w") as metadata_json_file:
            json.dump(respilens_metadata, metadata_json_file, indent = 4)
            
        logger.info("Success.")
        

def main():
    """
    Main execution function.
    """
    
    # Create CL options 
    parser = argparse.ArgumentParser(description = "Download CDC data.")
    parser.add_argument("--resource-id",
                       type = str,
                       help = "Unique identifier for resource; found in resource URL.")
    parser.add_argument("--output-path", 
                       type = str,
                       help = "Pathway to directory to save data to.")
    parser.add_argument("--replace-column-names",
                       action = "store_true",  
                       default = True, 
                       help = "Replace short-form column names with long-form.")
    parser.add_argument("--dont-replace-column-names",
                       action = "store_false", 
                       dest = "replace_column_names",  
                       help = "Don't replace short-form column names with long-form.")
    parser.add_argument("--output-format",
                        nargs = '*',
                        choices = ['json', 'csv'],
                        required = False, 
                        help = "The format in which to save the data: 'json', 'csv', or both.")
    args = parser.parse_args()
    
    try:
        # Initialize the CDCData instance
        cdc_data = CDCData(args.resource_id, args.output_path) 
    
        # Store data and metadata from resource_id in a dictionary 
        data_and_metadata = cdc_data.download_cdc_data(args.replace_column_names)

        non_RL_formatted_location_dicts = []
        for region in data_and_metadata["data"].keys():
            try:
                validate(instance=data_and_metadata["data"][region], schema=RESPILENS_DATA_SCHEMA)
            except ValidationError as e:
                non_RL_formatted_location_dicts.append(data_and_metadata["data"][region])
        if non_RL_formatted_location_dicts:
            raise ValueError(f"Final JSON for locations {non_RL_formatted_location_dicts} failed RespiLens schema validation.")
    
        if args.output_format:
            # Save data locally, according to user input
            if "csv" in args.output_format: 
                cdc_data.save_data_csv(data_and_metadata["data_as_DF"], data_and_metadata["respilens_metadata"])
            if "json" in args.output_format:
                logger.info(f"Saving data as json to {args.output_path}.")
                saved_files_count = 0
                unsaved_files = []
                for region, region_data in data_and_metadata["data"].items():
                    try:
                        save_data(region_data, args.output_path)
                        saved_files_count += 1
                    except (FileExistsError, ValueError) as e:
                        unsaved_files.append((region, e))
                
                # Save the single metadata file
                save_metadata(data_and_metadata["respilens_metadata"], args.output_path)

                # Report successes and failures
                logger.info(f"Successfully saved JSON data for {saved_files_count} regions.")
                if unsaved_files:
                    logger.warning(f"Failed to save JSON data for {len(unsaved_files)} regions:")
                    for region, error in unsaved_files:
                        logger.warning(f"  - Region: {region}, Reason: {error}")
            
        # Edge case
        else:
            raise argparse.ArgumentTypeError(f"--output-format must either be 'json' or 'csv', received {args.output_format}.")
            
    except Exception as e:
        logger.error(f"Failed to download and/or save CDC data: {str(e)}")
        raise    
    

if __name__ == "__main__":
    main()