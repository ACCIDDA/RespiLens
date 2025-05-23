import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FluSightPreprocessor:
    def __init__(self, base_path: str, output_path: str, demo_mode: bool = False):
        """Initialize preprocessor with paths and mode settings"""
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.demo_mode = demo_mode
        self.demo_models = ['UNC_IDD-influpaint', 'FluSight-ensemble']
        self.all_models = set()  # Add this line

        # Define paths
        self.model_output_path = self.base_path / "model-output"
        self.target_data_path = self.base_path / "target-data/target-hospital-admissions.csv"
        self.locations_path = self.base_path / "auxiliary-data/locations.csv"

        # Validate paths exist
        self._validate_paths()

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Cache for processed data
        self.locations_data = None
        self.ground_truth = None
        self.forecast_data = None

        # Cache file listings
        self.model_files = {
            model_dir.name: list(model_dir.glob("*.csv")) + list(model_dir.glob("*.parquet"))
            for model_dir in self.model_output_path.glob("*")
            if model_dir.is_dir()
        }

        # Add lock for thread safety
        self.forecast_data_lock = Lock()

    def _validate_paths(self):
        """Validate all required paths exist"""
        required_paths = {
            'base_path': self.base_path,
            'model_output_path': self.model_output_path,
            'target_data_path': self.target_data_path,
            'locations_path': self.locations_path
        }

        for name, path in required_paths.items():
            if not path.exists():
                raise ValueError(f"{name} does not exist: {path}")

    def load_locations(self) -> pd.DataFrame:
        """Load and cache locations data"""
        if self.locations_data is None:
            logger.info("Loading locations data...")
            self.locations_data = pd.read_csv(self.locations_path)
        return self.locations_data

    def load_ground_truth(self) -> Dict:
        """Load and process ground truth data"""
        if self.ground_truth is None:
            logger.info("Loading ground truth data...")
            df = pd.read_csv(self.target_data_path)
            df['date'] = pd.to_datetime(df['date'])

            # Filter to relevant dates and sort
            df = df[df['date'] >= pd.Timestamp('2023-10-01')].sort_values('date')
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

            # Create optimized structure for visualization
            self.ground_truth = {}
            for location in df['location'].unique():
                loc_data = df[df['location'] == location]
                self.ground_truth[location] = {
                    'dates': loc_data['date_str'].tolist(),
                    'values': loc_data['value'].tolist(),
                    'rates': loc_data['weekly_rate'].tolist()
                }

        return self.ground_truth

    def read_model_outputs(self) -> Dict:
        """Read and process all model output files efficiently"""
        if self.forecast_data is not None:
            return self.forecast_data

        logger.info("Reading model output files...")
        self.forecast_data = {}

        # Get list of model directories
        model_dirs = [d for d in self.model_output_path.glob("*") if d.is_dir()]
        if self.demo_mode:
            model_dirs = [d for d in model_dirs if d.name in self.demo_models]

        # Add model list
        for model_dir in model_dirs:
            self.all_models.add(model_dir.name)

        def process_file(file_info):
            model_name, file_path = file_info
            try:
                # Read file based on extension
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path, dtype={'location': str})  # Force location as string
                else:  # .parquet
                    df = pd.read_parquet(file_path)
                    df['location'] = df['location'].astype(str)  # Convert location to string after reading

                # Process dates
                # remove samples if any
                df = df[~df['output_type'].str.contains('sample')]
                df['reference_date'] = pd.to_datetime(df['reference_date'])
                if 'target_end_date' in df.columns:
                    df['target_end_date'] = pd.to_datetime(df['target_end_date'])

                # Create a local dict for this file's data
                processed_data = {}

                # Group by location and organize data
                for location, loc_group in df.groupby('location'):
                    if location not in processed_data:
                        processed_data[location] = {}

                    # Group by reference date
                    for ref_date, date_group in loc_group.groupby('reference_date'):
                        ref_date_str = ref_date.strftime('%Y-%m-%d')

                        if ref_date_str not in processed_data[location]:
                            processed_data[location][ref_date_str] = {}

                        # Group by target type
                        for target, target_group in date_group.groupby('target'):
                            if target not in processed_data[location][ref_date_str]:
                                processed_data[location][ref_date_str][target] = {}

                            # Store model predictions
                            model_data = self._process_model_predictions(target_group)
                            processed_data[location][ref_date_str][target][model_name] = model_data

                return model_name, file_path, processed_data
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                return None

        # Create list of work
        work_items = []
        for model_dir in model_dirs:
            model_name = model_dir.name
            files = self.model_files[model_name]
            work_items.extend([(model_name, f) for f in files])

        # Add progress tracking
        total_files = sum(len(files) for files in self.model_files.values())
        logger.info(f"Processing {total_files} files across {len(self.model_files)} models")

        # Process in parallel
        with tqdm(total=total_files, desc="Reading files") as pbar:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_file, item) for item in work_items]
                for future in as_completed(futures):
                    pbar.update(1)
                    result = future.result()
                    if result:
                        with self.forecast_data_lock:
                            model_name, file_path, processed_data = result
                            # Merge processed_data into self.forecast_data
                            for location, location_data in processed_data.items():
                                if location not in self.forecast_data:
                                    self.forecast_data[location] = {}
                                # Deep merge the data
                                for date, date_data in location_data.items():
                                    self.forecast_data[location][date] = self.forecast_data[location].get(date, {})
                                    for target, target_data in date_data.items():
                                        self.forecast_data[location][date][target] = self.forecast_data[location][date].get(target, {})
                                        self.forecast_data[location][date][target].update(target_data)

        return self.forecast_data

    def _process_model_predictions(self, group_df: pd.DataFrame) -> Dict:
        """Process model predictions into an optimized format for visualization"""
        output_type = group_df['output_type'].iloc[0]

        if output_type == 'quantile':
            # For quantiles, create a structure optimized for plotting
            predictions = {}
            for horizon, horizon_df in group_df.groupby('horizon'):
                horizon_df = horizon_df.sort_values('output_type_id')
                predictions[str(int(horizon))] = {
                    'date': horizon_df['target_end_date'].iloc[0].strftime('%Y-%m-%d'),
                    'quantiles': horizon_df['output_type_id'].astype(float).tolist(),
                    'values': horizon_df['value'].tolist()
                }
            return {'type': 'quantile', 'predictions': predictions}

        elif output_type == 'pmf':
            # For probability mass functions
            predictions = {}
            for horizon, horizon_df in group_df.groupby('horizon'):
                predictions[str(int(horizon))] = {
                    'date': horizon_df['target_end_date'].iloc[0].strftime('%Y-%m-%d'),
                    'categories': horizon_df['output_type_id'].tolist(),
                    'probabilities': horizon_df['value'].tolist()
                }
            return {'type': 'pmf', 'predictions': predictions}

        else:  # sample
            predictions = {}
            for horizon, horizon_df in group_df.groupby('horizon'):
                predictions[str(int(horizon))] = {
                    'date': horizon_df['target_end_date'].iloc[0].strftime('%Y-%m-%d'),
                    'samples': horizon_df['value'].tolist()
                }
            return {'type': 'sample', 'predictions': predictions}

    def create_visualization_payloads(self):
        """Create optimized payloads for visualization"""
        logger.info("Creating visualization payloads...")

        # Load required data
        locations = self.load_locations()
        ground_truth = self.load_ground_truth()
        forecast_data = self.read_model_outputs()

        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Save metadata about available models
        metadata = {
            "shortName": "flusight",
            "fullName": "FluSight Hospitalisation Forecasts",
            "defaultView": "detailed",
            "targets": ["wk inc flu hosp"], # Primary target, other seasonal/peak targets might be present in data
            "quantile_levels": [
                0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
            ], # Standard 23 quantiles
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': sorted(list(self.all_models)),  # Keep global list here
            'locations': [
                {
                    'location': str(row.location),
                    'abbreviation': str(row.abbreviation),
                    'location_name': str(row.location_name),
                    'population': float(row.population)
                }
                for _, row in locations.iterrows()
                if pd.notna(row.location_name) and pd.notna(row.abbreviation)
            ],
            'demo_mode': self.demo_mode
        }

        # Path for dataset metadata
        dataset_metadata_path = self.output_path / "datasets" / "flusight"
        dataset_metadata_path.mkdir(parents=True, exist_ok=True)

        with open(dataset_metadata_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Path for location-specific projection files
        payload_path = self.output_path / "datasets" / "flusight" / "projections"
        payload_path.mkdir(parents=True, exist_ok=True)

        # Create and save location-specific payloads
        for _, location_info in tqdm(locations.iterrows(), desc="Creating location payloads"):
            location = location_info['location']
            if location == '06':  # California's FIPS code
                models = [model for date_data in forecast_data.get('06', {}).values()
                          for target_data in date_data.values()
                          for model in target_data.keys()]
                logger.info(f"CA forecast data models: {models}")
            # Convert pandas Series to dict first
            metadata_dict = {
                'location': str(location_info['location']),
                'abbreviation': str(location_info['abbreviation']),
                'location_name': str(location_info['location_name']),
                'population': float(location_info['population'])
            }

            # Before the payload creation, get location-specific models
            location_models = set()
            if location in forecast_data:
                for date_data in forecast_data[location].values():
                    for target_data in date_data.values():
                        location_models.update(target_data.keys())

            payload = {
                'metadata': metadata_dict,
                'ground_truth': {
                    'dates': ground_truth.get(location, {'dates': []})['dates'],
                    'values': [None if pd.isna(x) else x for x in ground_truth.get(location, {'values': []})['values']],
                    'rates': [None if pd.isna(x) else x for x in ground_truth.get(location, {'rates': []})['rates']]
                },
                'forecasts': forecast_data.get(location, {}),
                'available_models': sorted(list(location_models)),  # Location-specific models
                'all_models': sorted(list(self.all_models))  # Add global model list here too
            }

            # Save location payload with abbreviation in filename
            # Normalize the location abbreviation and remove any whitespace
            location_abbrev = str(location_info['abbreviation']).strip()
            if not location_abbrev:
                continue  # Skip if no valid abbreviation
            with open(payload_path / f"{location_abbrev}.json", 'w') as f:
                json.dump(payload, f, cls=NpEncoder)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(description='Process FluSight forecast data for visualization')
    parser.add_argument('--hub-path', type=str, default='./FluSight-forecast-hub',
                      help='Path to FluSight forecast hub repository')
    parser.add_argument('--output-path', type=str, default='./processed_data',
                      help='Path for output files')
    parser.add_argument('--demo', action='store_true',
                      help='Run in demo mode with only UNC_IDD-influpaint and FluSight-ensemble models')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Set logging level')

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(args.log_level)

    try:
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir()}")
        logger.info(f"Starting preprocessing with hub path: {args.hub_path}")
        logger.info(f"Output path: {args.output_path}")
        logger.info(f"Demo mode: {args.demo}")

        preprocessor = FluSightPreprocessor(args.hub_path, args.output_path, args.demo)
        preprocessor.create_visualization_payloads()

        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"Failed to run preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
