import argparse
import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_flusight_data(input_file: str, output_path: str, locations_path: str):
    """Process Flusight target-hospital-admissions data and create metafiles."""
    try:
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)

        # Convert location column to strings
        df['location'] = df['location'].astype(str)

        # Load locations
        logger.info(f"Loading locations data from {locations_path}")
        locations = pd.read_csv(locations_path)
        location_map = dict(zip(locations['abbreviation'].str.upper(), locations.to_dict('records')))

        # Create output directories
        output_path = Path(output_path)
        target_dir = output_path / "nhsn"
        target_dir.mkdir(parents=True, exist_ok=True)
        app_public_dir = Path("app/public/processed_data/nhsn")
        app_public_dir.mkdir(parents=True, exist_ok=True)
        
        print(locations)

        # Process each location
        for location in df['location'].unique():
            # print(location_array)
            # location = location_array[0] #extract the string
            logger.info(f"Processing location: {location}")

            loc_data = df[df['location'] == location].sort_values('date')

            # Find matching location in locations data
            matching_location = None
            for _, row in locations.iterrows():
                if row['abbreviation'].upper() == location.upper() or row['location'] == location:
                    matching_location = row.to_dict()
                    break

            if matching_location:
                # Use location info from locations.csv
                metadata = {
                    "location": matching_location['location'],
                    "abbreviation": matching_location['abbreviation'],
                    "location_name": matching_location['location_name'],
                    "population": float(matching_location['population'])
                }
            else:
                # Fallback to Flusight location if no match
                logger.warning(f"Location '{location}' not found in locations file.")
                metadata = {
                    "location": location,
                    "abbreviation": location,
                    "location_name": location,
                    "population": 0.0
                }

            # Process data
            values = []
            dates = []
            for _, row in loc_data.iterrows():
                try:
                    values.append(float(row['value']))
                    dates.append(pd.to_datetime(row['date']).strftime('%Y-%m-%d'))
                except (ValueError, TypeError):
                    values.append(None)
                    dates.append(pd.to_datetime(row['date']).strftime('%Y-%m-%d'))

            # Create location data structure
            location_data = {
                'metadata': metadata,
                'ground_truth': {
                    'dates': dates,
                    'values': values
                },
                'data': {
                    'official': {},
                    'preliminary': {}
                }
            }

            # Save to JSON
            output_file = target_dir / f"{matching_location['abbreviation']}_nhsn.json"
            app_output_file = app_public_dir / f"{matching_location['abbreviation']}_nhsn.json"

            logger.info(f"Saving data to {output_file} and {app_output_file}")
            with open(output_file, 'w') as f:
                json.dump(location_data, f, indent=2)
            with open(app_output_file, 'w') as f:
                json.dump(location_data, f, indent=2)

        logger.info("Flusight data processing complete!")

    except Exception as e:
        logger.error(f"Failed to process Flusight data: {str(e)}")
        raise

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Process Flusight target-hospital-admissions data')
    parser.add_argument('--input-file', type=str, default='FluSight-forecast-hub/target-data/target-hospital-admissions.csv',
                      help='Path to input CSV file')
    parser.add_argument('--output-path', type=str, default='./processed_data',
                      help='Path for output files')
    parser.add_argument('--locations-path', type=str, default='locations.csv',
                      help='Path to locations.csv file')
    args = parser.parse_args()

    try:
        process_flusight_data(args.input_file, args.output_path, args.locations_path)
    except Exception as e:
        logger.error(f"Failed to process Flusight data: {str(e)}")
        raise

if __name__ == "__main__":
    main()

