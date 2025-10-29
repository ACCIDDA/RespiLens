"""
Class to process CDC/NHSN data (pd.DataFrame) into RespiLens-style JSON output.

Data is processed from resource id 'ua7e-t2fy'.
"""

import logging
import numpy as np
import os
import pandas as pd
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from helper import NHSN_COLUMN_MASKS, get_location_info, LOCATIONS_MAP

logger = logging.getLogger(__name__)
script_dir = os.path.dirname(__file__) 
locations_file_path = os.path.join(script_dir, 'locations.csv')


LOCATIONS_ABBREV = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 
        'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 
        'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 
        'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'US'
    ]


class NHSNDataProcessor:
    def __init__(self, resource_id, replace_column_names: bool = True):
        self.replace_column_names = replace_column_names
        self.data_url = "https://data.cdc.gov/resource/" + f"{resource_id}.json"
        self.metadata_url = "https://data.cdc.gov/api/views/" + f"{resource_id}.json"
        self.output_dict = {}
        self.locations_data = pd.read_csv(locations_file_path)

        self._process_data()

    
    def _process_data(self):
        """Fetches, processes, and structures NHSN data into self.output_dict"""
        # Get data set up 
        logger.info(f"Retrieving NHSN data from {self.data_url}...")
        data = pd.DataFrame(self._retrieve_data_from_endpoint_aslist()) # read from endpoint
        # data = data[NHSN_COLUMN_MASKS['RAW_PATIENT_COUNTS']] # only include a specific subset of columns for plotting now 
        non_numeric_cols = ['jurisdiction', 'weekendingdate'] # make numeric cols not strings
        for col in data.columns:
            if col not in non_numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='raise')
        data = data.replace(np.nan, value=None) # cleanse NaN values 
        data.loc[data['jurisdiction'].str.lower() == 'usa', 'jurisdiction'] = 'US' # change USA jurisdiction to US
        data = data[data['jurisdiction'].isin(LOCATIONS_ABBREV)].copy() # filter out unwanted regions
        data['weekendingdate'] = pd.to_datetime(data['weekendingdate']).dt.strftime('%Y-%m-%d') # ensure date columns are dates
        # Get metadata set up
        cdc_metadata = (requests.get(self.metadata_url)).json() 
        logger.info("Success ✅")

        # Process the data
        # Pipeline #1: key on longform location column name
        logger.info("Processing NHSN data...")
        if self.replace_column_names: 
            data = self._replace_column_names(data, cdc_metadata) 
            self.output_dict["metadata.json"] = self._build_metadata_file(list(data.columns), list(set(data['Geographic aggregation'])))
            unique_regions = set(data['Geographic aggregation'])
            for region in unique_regions:
                current_region_fips_code = LOCATIONS_MAP[region]
                current_region_df = data[data['Geographic aggregation'] == region]
                current_region_df = current_region_df.sort_values(by='Week Ending Date')
                series = {
                    "dates": list(current_region_df['Week Ending Date'])
                }
                columns = [col for col in current_region_df.columns if col not in ["Geographic aggregation", "Week Ending Date"]]
                for column in columns:
                    series[column] = list(current_region_df[column])
                json_struct = {
                    "metadata": {
                        "location": current_region_fips_code,
                        "abbreviation": region,
                        "location_name": get_location_info(location_data=self.locations_data, location=current_region_fips_code, value_needed="location_name"),
                        "population": get_location_info(location_data=self.locations_data, location=current_region_fips_code, value_needed='population'),
                        "dataset": "NHSN",
                        "series_type": "timeseries"
                    },
                    "series": series
                }
                self.output_dict[f"{region}_nhsn.json"] = json_struct

        # Pipeline #2: key on shortform location column name
        else:
            self.output_dict["metadata.json"] = self._build_metadata_file(list(data.columns), list(set(data['jurisdiction'])))
            unique_regions = set(data['jurisdiction'])
            for region in unique_regions:
                current_region_df = data[data['jurisdiction'] == region]
                current_region_df = current_region_df.sort_values(by='weekendingdate')
                series = {
                    "dates": list(current_region_df['weekendingdate'])
                }
                columns = [col for col in current_region_df.columns if col not in ["jurisdiction", "weekendingdate"]]
                for column in columns:
                    series[column] = list(current_region_df[column])
                json_struct = {
                    "metadata": {
                        "location": region,
                        "abbreviation": "",
                        "location_name": "",
                        "population": 0.0,
                        "dataset": "NHSN",
                        "series_type": "time series"
                    },
                    "series": series
                }
                self.output_dict[f"{region}_nhsn.json"] = json_struct

        logger.info("Success ✅")
        

    def _retrieve_data_from_endpoint_aslist(self) -> list[dict]:
        """Downloads NHSN data from the endpoint with pagination and retries."""
        
        session = requests.Session()
        retries = Retry(total=5,
                        backoff_factor=1,
                        status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        all_data = []
        offset = 0
        batch_size = 1000
        while True:
            params = {"$limit": batch_size, "$offset": offset}
            try:
                # Use the configured session to make the request
                data_response = session.get(self.data_url, params=params, timeout=30)
                data_response.raise_for_status()
                batch_data = data_response.json()
                if not batch_data:
                    break
                all_data.extend(batch_data)
                offset += batch_size
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error downloading data: {str(e)}")
                raise
        return all_data


    def _replace_column_names(self, data: pd.DataFrame, cdc_metadata: dict) -> pd.DataFrame:
        """Replace short-form column names with long-form column names"""
        column_name_map = {
                col_info['fieldName']: col_info['name']
                for col_info in cdc_metadata['columns']
            }
        return data.rename(columns=column_name_map, errors="ignore")
    

    def _build_metadata_file(self, columns: list[str], locations: list[str]) -> dict:
        """Build a single output metadata.json file (one per dataset output)"""
        metadata_file_contents = {
            "last_updated": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "dataset": "NHSN",
            "columns": columns,
            "locations": locations
        }
        return metadata_file_contents
