import os
import pandas as pd
import json
import pyarrow  # Ensure this is installed with: pip install pyarrow
from pathlib import Path
import logging
from typing import Optional, Dict, List, Any
from pydantic import ValidationError
try:
    # Assuming schemas are accessible in the PYTHONPATH or via relative imports
    from schemas.dataset_metadata import RSVHubDatasetMetadata
    from schemas.projections import RSVLocationProjectionsFile
except ImportError:
    # Fallback for direct script execution if schemas is a sibling directory
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from schemas.dataset_metadata import RSVHubDatasetMetadata
    from schemas.projections import RSVLocationProjectionsFile
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RSVPreprocessor:
    def __init__(self, base_path: str, output_path: str, demo_mode: bool = False):
        """Initialize preprocessor with paths and mode settings"""
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.demo_mode = demo_mode
        self.all_models = set()  # Add this line

        # Define paths
        self.model_output_path = self.base_path / "model-output"
        # find the target data path: sort all *_rsnet_hospitalization.csv files by date and take the last one
        target_data_files = sorted(self.base_path.glob("target-data/*_rsvnet_hospitalization.csv"))
        if not target_data_files:
            raise ValueError("No target data files found")
        self.target_data_path = target_data_files[-1]

        #self.target_data_path = self.base_path / "target-data/2025-01-17_rsvnet_hospitalization.csv"
        self.locations_path = self.base_path / "auxiliary-data/location_census/locations.csv"

        # Validate paths exist
        self._validate_paths()

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Cache for processed data
        self.locations_data = None
        self.ground_truth = None
        self.forecast_data = None

        # Define accepted age groups
        self.age_groups = ["0-0.99", "1-4", "5-64", "65-130", "0-130"]

        # Cache file listings
        self.model_files = {
            model_dir.name: list(model_dir.glob("*.parquet"))
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
            self.locations_data = pd.read_csv(self.locations_path, dtype={'location': str})
        return self.locations_data

    def load_ground_truth(self) -> Dict:
        """Load and process ground truth data"""
        if self.ground_truth is None:
            logger.info("Loading ground truth data...")
            df = pd.read_csv(self.target_data_path, dtype={'location': str})

            # Filter only inc hosp rows and remove NA values
            df = df[df['target'] == 'inc hosp']
            df = df[df['value'].notna()]

            # Create mapping for age group aggregation
            age_group_mapping = {
                # 0-0.99 combines 0-0.49 and 0.5-0.99
                "0-0.99": ["0-0.49", "0.5-0.99"],
                # 1-4 combines 1-1.99 and 2-4
                "1-4": ["1-1.99", "2-4"],
                # 5-64 combines 5-17, 18-49, and 50-64
                "5-64": ["5-17", "18-49", "50-64"],
                # 65-130 is already correct (65+)
                "65-130": ["65-130"],
                # 0-130 is already correct (overall)
                "0-130": ["0-130"]
            }

            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= pd.Timestamp('2023-10-01')].sort_values('date')
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

            # Create optimized structure for visualization with aggregated age groups
            self.ground_truth = {}
            for location in df['location'].unique():
                loc_data = df[df['location'] == location]
                self.ground_truth[location] = {}

                # Process each target age group
                for target_group, source_groups in age_group_mapping.items():
                    # Filter data for the source age groups and aggregate
                    age_data = loc_data[loc_data['age_group'].isin(source_groups)]
                    if not age_data.empty:
                        # Sum values for the same date across source age groups
                        agg_data = age_data.groupby('date_str')['value'].sum().reset_index()
                        self.ground_truth[location][target_group] = {
                            'dates': agg_data['date_str'].tolist(),
                            'values': agg_data['value'].tolist()
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

        # Add model list
        for model_dir in model_dirs:
            self.all_models.add(model_dir.name)

        def process_file(file_info):
            model_name, file_path = file_info
            try:
                # Use engine='pyarrow' to ensure compatibility
                df = pd.read_parquet(file_path, engine='pyarrow')
                df['location'] = df['location'].astype(str)  # Convert location to string after reading

                # Add default model name if not present
                if 'model' not in df.columns:
                    df['model'] = model_name

                # Add origin_date if not present
                if 'origin_date' not in df.columns and 'forecast_date' in df.columns:
                    df['origin_date'] = pd.to_datetime(df['forecast_date'])

                # Ensure expected columns exist
                required_columns = ['location', 'origin_date', 'age_group', 'target', 'output_type', 'output_type_id', 'value']

                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    logger.warning(f"Missing columns in {file_path}: {missing_columns}")
                    return None

                # Process dates and filter
                df = df[~df['output_type'].str.contains('sample')]
                df['origin_date'] = pd.to_datetime(df['origin_date'])

                # Create a local dict for this file's data
                processed_data = {}

                # Group by location and organize data
                for location, loc_group in df.groupby('location'):
                    if location not in processed_data:
                        processed_data[location] = {}

                    # Group by origin date
                    for origin_date, date_group in loc_group.groupby('origin_date'):
                        origin_date_str = origin_date.strftime('%Y-%m-%d')

                        if origin_date_str not in processed_data[location]:
                            processed_data[location][origin_date_str] = {}

                        # Group by age group
                        for age_group, age_group_data in date_group.groupby('age_group'):
                            if age_group not in self.age_groups:
                                continue

                            if age_group not in processed_data[location][origin_date_str]:
                                processed_data[location][origin_date_str][age_group] = {}

                            # Group by target type
                            for target, target_group in age_group_data.groupby('target'):
                                if target not in processed_data[location][origin_date_str][age_group]:
                                    processed_data[location][origin_date_str][age_group][target] = {}

                                # Store model predictions
                                model_data = self._process_model_predictions(target_group)
                                processed_data[location][origin_date_str][age_group][target][model_name] = model_data

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
                                    for age_group, age_group_data in date_data.items():
                                        self.forecast_data[location][date][age_group] = self.forecast_data[location][date].get(age_group, {})
                                        for target, target_data in age_group_data.items():
                                            self.forecast_data[location][date][age_group][target] = self.forecast_data[location][date][age_group].get(target, {})
                                            self.forecast_data[location][date][age_group][target].update(target_data)

        return self.forecast_data

    def _process_model_predictions(self, group_df: pd.DataFrame) -> Dict:
        """Process model predictions into an optimized format for visualization"""
        output_type = group_df['output_type'].iloc[0]

        if output_type == 'quantile':
            # For quantiles, group by horizon first
            predictions = {}
            for horizon, horizon_df in group_df.groupby('horizon'):
                # Sort by quantile to ensure correct order
                horizon_df = horizon_df.sort_values('output_type_id')

                predictions[str(int(horizon))] = {
                    'quantiles': horizon_df['output_type_id'].astype(float).tolist(),
                    'values': horizon_df['value'].tolist(),
                    # Optional: include model name for additional context
                    'model': group_df['model'].iloc[0]
                }
            return {'type': 'quantile', 'predictions': predictions}

        else:  # sample
            predictions = {}
            for horizon, horizon_df in group_df.groupby('horizon'):
                predictions[str(int(horizon))] = {
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

        # Prepare dataset-level metadata
        dataset_metadata_dict: Dict[str, Any] = {
            "shortName": "rsv_hub",
            "fullName": "RSV Hospitalisation Forecasts",
            "defaultView": "detailed",
            "targets": ["inc hosp"], # Primary target for RSV forecasts
            "quantile_levels": [
                0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
            ], # Standard 23 quantiles
            'last_updated': pd.Timestamp.now(), # Pydantic will convert to ISO string
            'models': sorted(list(self.all_models)),
            'age_groups': self.age_groups,
            'locations': [
                {
                    'location': str(row.location),
                    'abbreviation': str(row.abbreviation),
                    'name': str(row.location_name), # Ensure 'name' matches LocationBase
                    'population': float(row.population) if pd.notna(row.population) else None
                }
                for _, row in locations.iterrows()
                if pd.notna(row.location_name) and pd.notna(row.abbreviation)
            ],
            'demo_mode': self.demo_mode
        }

        dataset_metadata_output_dir = self.output_path / "datasets" / "rsv_hub"
        dataset_metadata_output_dir.mkdir(parents=True, exist_ok=True)
        dataset_metadata_filename = dataset_metadata_output_dir / 'metadata.json'

        try:
            validated_dataset_metadata = RSVHubDatasetMetadata(**dataset_metadata_dict)
            with open(dataset_metadata_filename, 'w') as f:
                json.dump(validated_dataset_metadata.dict(by_alias=True), f, indent=2)
            logger.info(f"Successfully validated and saved dataset metadata to {dataset_metadata_filename}")
        except ValidationError as e:
            logger.error(f"Validation Error for dataset metadata {dataset_metadata_filename}: {e}")
            raise

        # Path for location-specific projection files
        projections_output_path = self.output_path / "datasets" / "rsv_hub" / "projections"
        projections_output_path.mkdir(parents=True, exist_ok=True)

        # Create and save location-specific payloads
        for _, location_info in tqdm(locations.iterrows(), desc="Creating location payloads"):
            current_location_fips = str(location_info['location'])
            location_abbrev = str(location_info['abbreviation']).strip()

            if not location_abbrev:
                logger.warning(f"Skipping location {current_location_fips} due to missing abbreviation.")
                continue
            
            output_filename = projections_output_path / f"{location_abbrev}.json"

            file_metadata_payload = {
                "dataset": "rsv_hub",
                "location": current_location_fips,
                "abbreviation": location_abbrev,
                "name": str(location_info['location_name']),
                "population": float(location_info['population']) if pd.notna(location_info['population']) else None
            }
            
            # Ensure forecasts key exists, even if empty, for schema validation
            current_location_forecasts = forecast_data.get(current_location_fips, {})

            payload_for_validation = {
                "metadata": file_metadata_payload,
                "forecasts": current_location_forecasts
            }

            try:
                validated_loc_projection = RSVLocationProjectionsFile(**payload_for_validation)
                with open(output_filename, 'w') as f:
                    # The original script did not use NpEncoder for RSV, Pydantic's .dict() should handle standard types.
                    json.dump(validated_loc_projection.dict(by_alias=True), f, indent=2)
                # logger.info(f"Successfully validated and saved projection for {location_abbrev} to {output_filename}")
            except ValidationError as e:
                logger.error(f"Validation Error for projection {output_filename}: {e}")
                raise

def main():
    parser = argparse.ArgumentParser(description='Process RSV forecast data for visualization')
    parser.add_argument('--hub-path', type=str, default='./rsv-forecast-hub',
                      help='Path to RSV forecast hub repository')
    parser.add_argument('--output-path', type=str, default='./processed_data',
                      help='Path for output files')
    parser.add_argument('--demo', action='store_true',
                      help='Run in demo mode')
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

        preprocessor = RSVPreprocessor(args.hub_path, args.output_path, args.demo)
        preprocessor.create_visualization_payloads()

        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"Failed to run preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
