"""
module-level docs
"""

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
    """
    A class to retrieve and store data + metadata for a specified CDC resource.
    """
    def __init__(self, resource_id: str):
        """
        Initialize the CDCData class.
        
        Args:
            resource_id: The unique resource identifier string.
        """
        self.data_url = "https://data.cdc.gov/resource/" + f"{resource_id}.json"
        self.metadata_url = "https://data.dcd.gov/api/views/" + f"{resource_id}.json"
    
    def download_cdc_data(self, replace_column_names: bool = True) -> dict: # return dict with df and dict of metadata info
        """
        Retrieves and stores data + metadata for specified resource_id.
        
        Arg:
            replace_column_names: Flag to replace short column names with long names.
        
        Returns:
            A dictionary with;
                'data' storing the pd.DataFrame of data, and
                'metadata' storing a dictionary of metadata.
        """
        
        # Initialize dictionary to hold output
        output = {}
        
        
        # Retrieve metadata from endpoint
        logger.info(f"Retrieving metadata from {self.metadata_url}.")
        metadata = self.download_metadata_from_endpoint()
        
        # Retrieve data from endpoint and convert to DataFrame
        logger.info(f"Retrieving data from {self.data_url}.")
        data_list = self.download_data_from_endpoint()
        data_df = pd.DataFrame(data_list)
        
        if replace_column_names:
            data_df = self.replace_column_names(data_df, metadata)
        
        # Add DataFrame and metadata to output dict
        output['metadata'] = metadata  
        output['data'] = data_df
        
        return output
    
    def downlod_data_from_endpoint(self) -> list[dict]:
        """
        Download CDC data from a given endpoint with pagination.
            
        Returns:
            A list of dictionaries containing the json data for specified resource.  
        """
        all_data = []
        offset = 0
        batch_size = 1000
        
        # Reqeuest data from API in batches
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
    
    def download_metadata_from_endpoint(self) -> dict: 
        """
        Downlaod CDC metadata from a given endpoint.
        
        Returns:
            A dictionary of the metadata for specified resource.
        """
        
        # Retrieve metadata from endpoint 
        metadata_response = requests.get(self.metadata_url)
        metadata = metadata_response.json()
        
        return metadata
    
    def replace_column_names(data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Replace short-form column names with long-form column names.
        
        Args:
            data: A pd.DataFrame of the data
            metadat: A dictionary of metadata containing column name information
            
        Returns:
            A pd.DataFrame with long-form column names.
        """
        pass # To be written

