"""
This module facilitates the retrieval, transformation, and storage of data from the CDC's public API.

Data can either be stored as raw json, or as a pd.DataFrame. Metdata is always stored as raw json.

Classes:
    CDCData: Handles downloading, transforming, and saving data from a specific CDC dataset resource.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests


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
    
    def download_cdc_data(self, output_format: str, replace_column_names: bool = True) -> dict: 
        """
        Retrieves and stores data + metadata for specified resource_id.
        
        Arg:
            replace_column_names: Flag to replace short column names with long names.
            output_format: Store/save data as json/.json or as a pd.DataFrame/.csv.
        
        Returns:
            A dictionary with;
                'data' storing data as a pd.Dataframe or a list, and
                'metadata' storing metadata as a dictionary of raw json.
        """
        
        # Initialize dictionary to hold output
        output = {}
        
        # Retrieve metadata from endpoint
        logger.info(f"Retrieving metadata from {self.metadata_url}.")
        metadata_response = requests.get(self.metadata_url)
        metadata = metadata_response.json()
          
        # Retrieve data from endpoint
        # If we want to save data as csv:
        if output_format.lower() == 'csv':
            logger.info(f"Retrieving data from {self.data_url}.")
            data_list = self.retrieve_data_from_endpoint_aslist()
            data = pd.DataFrame(data_list)
            if replace_column_names:
                data = self.replace_column_names(data, metadata)
        # If we want to save data as raw json:
        elif output_format.lower() == 'json':
            # Retrieve data from endpoint and parse as JSON
            logger.info(f"Retrieving data from {self.data_url}.")
            data_response = requests.get(self.data_url)
            data = data_response.json()
            if replace_column_names:
                data = self.replace_column_names(data, metadata)
        
        # Edge case
        else:
            raise argparse.ArgumentTypeError(f"--output-format must either be 'json' or 'csv', received {args.output_format}.")
                    
        # Add DataFrame and metadata to output dict
        output['metadata'] = metadata  
        output['data'] = data
        
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
    
    def replace_column_names(self, data: pd.DataFrame | dict, metadata: dict) -> pd.DataFrame | dict:
        """
        Replace short-form column names with long-form column names.
        
        Args:
            data: A pd.DataFrame or list (json format) of the data
            metadata: A dictionary of metadata containing column name information
            
        Returns:
            A pd.DataFrame or dict (json) with long-form column names.
        """
        
        # Pull long column names from metadata
        long_column_names = []
        for column_index in range(len(metadata['columns'])):
            long_column_names.append(metadata['columns'][column_index]['name'])
            
        # Column replacement for a pd.DataFrame
        if isinstance(data, pd.DataFrame):
            data.columns = long_column_names
            return data
        
        # Column replacement for raw json
        elif isinstance(data, (dict, list)):
            if isinstance(data, dict):
                new_data = {}
                for i, (short_name, value) in enumerate(data.items()):
                    new_data[long_column_names[i]] = value
                return new_data
            else:
                new_data = []
                for record in data:
                    new_record = {}
                    for i, (short_name, value) in enumerate(record.items()):
                        new_record[long_column_names[i]] = value
                    new_data.append(new_record)
                return new_data
        
        # Edge case
        else:
            raise TypeError("data input must either be pd.DataFrame or dict.")
    
    def save_data_csv(self, data: pd.DataFrame, metadata: dict) -> None:
        """
        A method to save data at specified output_path as a CSV.
        
        Args:
            data: A pd.DataFrame containing the data.
            metadata: A dict containing the metadata.
        """
        
        # Create directories
        directory = self.output_path / f"cdc_{self.resource_id}"
        directory.mkdir(parents = True, exist_ok = True)
        logger.info(f"Saving data as CSV to {directory}...")
        
        # Save data to CSV, metadata to json
        data.to_csv(f"{directory}/data.csv", index=False)
        with open(f"{directory}/metadata.json", "w") as metadata_json_file:
            json.dump(metadata, metadata_json_file, indent = 4)
            
        logger.info("Success.")
    
    def save_data_json(self, data: list, metadata: dict) -> None:  
        """
        A method to save data at specified output_path as a json.
        
        Args:
            data: A list containing the data in raw json format.
            metadata: A dict containing the metadata in raw json format. 
        """
        
        # Create directories
        directory = self.output_path / f"cdc_{self.resource_id}"
        directory.mkdir(parents = True, exist_ok = True)
        logger.info(f"Saving data as json to {directory}...")
        
        # Save data and metadata to json
        with open(f"{directory}/data.json", "w") as data_json_file:
            json.dump(data, data_json_file, indent = 4)
        with open(f"{directory}/metadata.json", "w") as metadata_json_file:
            json.dump(metadata, metadata_json_file, indent = 4)
            
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
                        choices = ['json', 'csv'],
                        required = True, 
                        help = "The format in which to save the data: either 'json' or 'csv'.")
    args = parser.parse_args()
    
    try:
        # Initialize the CDCData instance
        cdc_data = CDCData(args.resource_id, args.output_path)
    
        # Store data and metadata from resource_id in a dictionary 
        data_and_metadata = cdc_data.download_cdc_data(args.output_format, args.replace_column_names)
    
        # Save data locally, according to user input
        if args.output_format == 'csv':
            cdc_data.save_data_csv(data_and_metadata["data"], data_and_metadata["metadata"])
        elif args.output_format == 'json':
            cdc_data.save_data_json(data_and_metadata["data"], data_and_metadata["metadata"])
        # Edge case
        else:
            raise argparse.ArgumentTypeError(f"--output-format must either be 'json' or 'csv', received {args.output_format}.")
            
    except Exception as e:
        logger.error(f"Failed to download and/or save CDC data: {str(e)}")
        raise 


if __name__ == "__main__":
    main()