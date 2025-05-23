import argparse
import pandas as pd
import requests
import json
from pathlib import Path
import logging
from datetime import datetime
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NHSNDataDownloader:
    def __init__(self, output_path: str, locations_path: Optional[str] = None):
        """Initialize the NHSN data downloader"""
        self.official_url = "https://data.cdc.gov/resource/ua7e-t2fy.json"
        self.preliminary_url = "https://data.cdc.gov/resource/mpgq-jmmr.json"
        self.output_path = Path(output_path)
        self.locations_path = Path(locations_path) if locations_path else None
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.locations_data = None

    def download_data(self, batch_size: int = 1000) -> pd.DataFrame:
        """Download all NHSN data using pagination from both endpoints"""
        logger.info("Starting NHSN data download...")

        # Download from both endpoints
        official_data = self._download_from_endpoint(self.official_url, batch_size, "official")
        preliminary_data = self._download_from_endpoint(self.preliminary_url, batch_size, "preliminary")

        # Convert both to dataframes
        official_df = pd.DataFrame(official_data)
        preliminary_df = pd.DataFrame(preliminary_data)

        # Debug logging
        logger.info(f"Official data shape: {official_df.shape}")
        logger.info(f"Official data columns: {official_df.columns.tolist()}")
        logger.info(f"Preliminary data shape: {preliminary_df.shape}")
        logger.info(f"Preliminary data columns: {preliminary_df.columns.tolist()}")

        # Combine the dataframes
        df = pd.concat([preliminary_df, official_df], ignore_index=True)
        logger.info(f"Combined data shape: {df.shape}")
        logger.info(f"Combined data columns: {df.columns.tolist()}")
        logger.info(f"Data types: {df['_type'].unique().tolist()}")

        return df

    def _download_from_endpoint(self, url: str, batch_size: int, data_type: str) -> list:
        """Download data from a specific endpoint"""
        all_data = []
        offset = 0

        while True:
            logger.info(f"Downloading {data_type} records {offset} to {offset + batch_size}")
            params = {
                "$limit": batch_size,
                "$offset": offset
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                batch_data = response.json()

                if not batch_data:  # No more data
                    break

                # Add _type field to each record
                for record in batch_data:
                    record['_type'] = data_type

                all_data.extend(batch_data)
                offset += batch_size
                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error downloading {data_type} data: {str(e)}")
                break

        return all_data

    def load_locations(self) -> pd.DataFrame:
        """Load and cache locations data"""
        if self.locations_data is None:
            logger.info("Loading locations data...")
            self.locations_data = pd.read_csv(self.locations_path)
        return self.locations_data

    def process_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process the downloaded data into official and preliminary dataframes"""
        logger.info("Processing NHSN data...")
        logger.info(f"Input DataFrame shape: {df.shape}")

        # Load locations for validation and special cases
        locations = self.load_locations()
        logger.info(f"Loaded locations data shape: {locations.shape}")
        valid_locations = set(locations['abbreviation'].str.upper())
        logger.info(f"Valid locations: {valid_locations}")

        # Create location column with special case handling
        mapping_dict = {
            'USA': 'US',  # Convert USA to US
            'Region 1': None,  # Filter out regions
            'Region 2': None,
            'Region 3': None,
            'Region 4': None,
            'Region 5': None,
            'Region 6': None,
            'Region 7': None,
            'Region 8': None,
            'Region 9': None,
            'Region 10': None,
            'GU': None,  # Filter out territories if needed
            'PR': None,
            'VI': None,
            'AS': None,
            'MP': None
        }

        # Split into official and preliminary dataframes
        official_df = df[df['_type'] == 'official'].copy()
        preliminary_df = df[df['_type'] == 'preliminary'].copy()

        # Process each dataframe
        for df in [official_df, preliminary_df]:
            # Map locations
            df['location'] = df['jurisdiction'].apply(lambda x: mapping_dict.get(x, x))

            # Drop rows where location is None (regions and territories)
            df.dropna(subset=['location'], inplace=True)

            # Rename date column and convert to datetime
            df.rename(columns={'weekendingdate': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

            # Convert numeric columns (exclude non-numeric columns)
            exclude_cols = ['location', 'jurisdiction', 'date', '_type']
            numeric_columns = [col for col in df.columns if col not in exclude_cols]

            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

            # Sort data
            df.sort_values(['date', 'location'], inplace=True)

        return official_df, preliminary_df

    def save_data(self, official_df: pd.DataFrame, preliminary_df: pd.DataFrame):
        """Save the processed data in a format compatible with RSV/Flu views"""
        logger.info("Starting save_data...")

        # Define and create base directories for dataset metadata
        dataset_metadata_base_output_path = self.output_path / "datasets" / "nhsn"
        dataset_metadata_base_output_path.mkdir(parents=True, exist_ok=True)
        dataset_metadata_base_app_public_path = Path("app/public/processed_data/datasets/nhsn")
        dataset_metadata_base_app_public_path.mkdir(parents=True, exist_ok=True)

        # Create dataset metadata content
        nhsn_metadata = {
            "shortName": "nhsn",
            "fullName": "National Healthcare Safety Network Data",
            "description": "Timeseries data for various health metrics reported by NHSN.",
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "columns_description_url": "https://www.cdc.gov/nhsn/covid19/report-patient-impact.html#anchor_1613680270822" # Example URL
        }

        # Define full paths for metadata files
        dataset_metadata_output_file = dataset_metadata_base_output_path / "metadata.json"
        dataset_metadata_app_public_file = dataset_metadata_base_app_public_path / "metadata.json"

        # Save dataset metadata
        with open(dataset_metadata_output_file, 'w') as f:
            json.dump(nhsn_metadata, f, indent=2)
        with open(dataset_metadata_app_public_file, 'w') as f:
            json.dump(nhsn_metadata, f, indent=2)
        logger.info(f"Saved NHSN dataset metadata to {dataset_metadata_output_file} and {dataset_metadata_app_public_file}")

        # Define and create directories for timeseries data
        target_dir = self.output_path / "datasets" / "nhsn" / "timeseries"
        target_dir.mkdir(parents=True, exist_ok=True)
        app_public_dir = Path("app/public/processed_data/datasets/nhsn/timeseries")
        app_public_dir.mkdir(parents=True, exist_ok=True)

        # Load locations for metadata
        locations = self.load_locations()
        location_map = dict(zip(locations['abbreviation'].str.upper(), locations.to_dict('records')))

        # Get all valid locations from both dataframes
        valid_locations = set(official_df['location'].unique()) | set(preliminary_df['location'].unique())

        # Create location-specific JSON files
        for location in tqdm(valid_locations, desc="Saving location data"):
            logger.info(f"Processing location: {location}")

            try:
                # Get location data from both dataframes
                official_loc = official_df[official_df['location'] == location].sort_values('date')
                preliminary_loc = preliminary_df[preliminary_df['location'] == location].sort_values('date')

                if official_loc.empty and preliminary_loc.empty:
                    logger.warning(f"No data for location {location}")
                    continue

                # Get location metadata
                loc_info = location_map.get(location, {})

                # Get all columns except metadata columns
                exclude_cols = ['location', 'jurisdiction', 'date', '_type']

                # Process official data
                official_columns = {}
                for col in official_loc.columns:
                    if col not in exclude_cols and official_loc[col].notna().any():
                        values = []
                        for v in official_loc[col].tolist():
                            if pd.isna(v) or v == 'NaN':
                                values.append(None)
                            else:
                                try:
                                    values.append(float(v))
                                except (ValueError, TypeError):
                                    values.append(None)
                        if any(v is not None for v in values):
                            official_columns[col] = values

                # Process preliminary data
                preliminary_columns = {}
                for col in preliminary_loc.columns:
                    if col not in exclude_cols and preliminary_loc[col].notna().any():
                        values = []
                        for v in preliminary_loc[col].tolist():
                            if pd.isna(v) or v == 'NaN':
                                values.append(None)
                            else:
                                try:
                                    values.append(float(v))
                                except (ValueError, TypeError):
                                    values.append(None)
                        if any(v is not None for v in values):
                            preliminary_columns[col] = values

                final_location_data = {}

                # Process official data
                if not official_loc.empty and official_columns:
                    official_dates = official_loc['date'].dt.strftime('%Y-%m-%d').tolist()
                    official_payload = {
                        "metadata": {
                            "dataset": "nhsn",
                            "location": location,
                            "location_name": loc_info.get('location_name', location),
                            "population": float(loc_info.get('population', 0)) if loc_info.get('population') is not None else None,
                            "series_type": "official"
                        },
                        "series": {
                            "dates": official_dates,
                            "columns": official_columns
                        }
                    }
                    final_location_data["official"] = official_payload

                # Process preliminary data
                if not preliminary_loc.empty and preliminary_columns:
                    preliminary_dates = preliminary_loc['date'].dt.strftime('%Y-%m-%d').tolist()
                    preliminary_payload = {
                        "metadata": {
                            "dataset": "nhsn",
                            "location": location,
                            "location_name": loc_info.get('location_name', location),
                            "population": float(loc_info.get('population', 0)) if loc_info.get('population') is not None else None,
                            "series_type": "preliminary"
                        },
                        "series": {
                            "dates": preliminary_dates,
                            "columns": preliminary_columns
                        }
                    }
                    final_location_data["preliminary"] = preliminary_payload

                # Save to both locations with new filename format
                if final_location_data:
                    output_file = target_dir / f"{location}.json"
                    app_output_file = app_public_dir / f"{location}.json"

                    logger.info(f"Saving data to {output_file} and {app_output_file}")

                    with open(output_file, 'w') as f:
                        json.dump(final_location_data, f, indent=2)
                    with open(app_output_file, 'w') as f:
                        json.dump(final_location_data, f, indent=2)
                else:
                    logger.warning(f"No data (official or preliminary) to save for location {location}")

            except Exception as e:
                logger.error(f"Error processing location {location}: {str(e)}")
                continue

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Download NHSN data')
    parser.add_argument('--output-path', type=str, default='./processed_data',
                      help='Path for output files')
    parser.add_argument('--locations-path', type=str,
                      help='Path to locations.csv file')
    args = parser.parse_args()

    try:
        downloader = NHSNDataDownloader(args.output_path, args.locations_path)

        # Download data
        df = downloader.download_data()

        # Process data - now returns two dataframes
        official_df, preliminary_df = downloader.process_data(df)

        # Save data with both dataframes
        downloader.save_data(official_df, preliminary_df)

        logger.info("NHSN data download and processing complete!")

    except Exception as e:
        logger.error(f"Failed to download and process NHSN data: {str(e)}")
        raise

if __name__ == "__main__":
    main()
