import argparse
import json
import logging
from pathlib import Path
import sys
from pydantic import ValidationError

# Allow imports from the 'schemas' directory, assuming 'scripts' is the parent of 'schemas'
# and this script is in 'scripts'.
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from schemas.global_metadata import GlobalMetadataFile, LocationsFile
    from schemas.dataset_metadata import FluSightDatasetMetadata, RSVHubDatasetMetadata, NHSNDatasetMetadata
    from schemas.projections import FluSightLocationProjectionsFile, RSVLocationProjectionsFile
    from schemas.timeseries import NHSNLocationTimeseriesFile
except ImportError as e:
    print(f"Error importing schemas: {e}. Ensure that the 'schemas' directory is in the correct location and PYTHONPATH is set if necessary.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

SCHEMA_DISPATCH = {
    "global_metadata": GlobalMetadataFile,
    "locations": LocationsFile,
    "flusight": {
        "dataset_metadata": FluSightDatasetMetadata,
        "projections": FluSightLocationProjectionsFile,
    },
    "rsv_hub": { # Corrected from rsv to rsv_hub to match directory names
        "dataset_metadata": RSVHubDatasetMetadata,
        "projections": RSVLocationProjectionsFile,
    },
    "nhsn": {
        "dataset_metadata": NHSNDatasetMetadata,
        "timeseries": NHSNLocationTimeseriesFile,
    }
}

error_count = 0
validated_file_count = 0

def validate_file(filepath: Path, model_cls, file_description: str):
    global error_count, validated_file_count
    
    if not filepath.exists():
        logging.warning(f"SKIPPED: {file_description} - File not found: {filepath}")
        return

    logging.info(f"VALIDATING: {file_description} - {filepath}")
    validated_file_count += 1
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        model_cls(**data)
        logging.info(f"OK: {filepath}")
    except json.JSONDecodeError as e:
        logging.error(f"ERROR: {filepath} - Invalid JSON: {e}")
        error_count += 1
    except ValidationError as e:
        logging.error(f"ERROR: {filepath} - Schema validation failed:\n{e}")
        error_count += 1
    except Exception as e:
        logging.error(f"ERROR: {filepath} - An unexpected error occurred: {e}")
        error_count += 1

def main():
    global error_count, validated_file_count

    parser = argparse.ArgumentParser(description="Validate processed Hubverse JSON data.")
    parser.add_argument("base_path", type=Path, help="Base path to the processed_data directory (e.g., app/public/processed_data)")
    args = parser.parse_args()

    base_path: Path = args.base_path

    if not base_path.is_dir():
        logging.error(f"Specified base path {base_path} is not a directory or does not exist.")
        sys.exit(1)

    # --- Validate global files ---
    logging.info("--- Validating Global Files ---")
    global_metadata_path = base_path / "metadata.json"
    validate_file(global_metadata_path, SCHEMA_DISPATCH["global_metadata"], "Global metadata.json")

    locations_path = base_path / "locations.json"
    validate_file(locations_path, SCHEMA_DISPATCH["locations"], "Global locations.json")

    # --- Validate dataset-specific files ---
    logging.info("\n--- Validating Dataset-Specific Files ---")
    datasets_path = base_path / "datasets"
    if not datasets_path.is_dir():
        logging.warning(f"SKIPPED: Datasets directory not found: {datasets_path}")
    else:
        for dataset_dir in datasets_path.iterdir():
            if not dataset_dir.is_dir():
                logging.warning(f"Skipping non-directory item in datasets: {dataset_dir.name}")
                continue
            
            dataset_name = dataset_dir.name
            logging.info(f"\nProcessing dataset: {dataset_name}")

            dataset_schemas = SCHEMA_DISPATCH.get(dataset_name)
            if not dataset_schemas:
                logging.warning(f"SKIPPED: No schemas defined for dataset '{dataset_name}' in {dataset_dir}")
                continue

            # Validate dataset metadata.json
            dataset_metadata_file = dataset_dir / "metadata.json"
            if dataset_schemas.get("dataset_metadata"):
                validate_file(dataset_metadata_file, dataset_schemas["dataset_metadata"], f"Dataset metadata for {dataset_name}")
            elif dataset_metadata_file.exists():
                 logging.warning(f"SKIPPED: Found dataset metadata.json for '{dataset_name}' but no schema defined for it: {dataset_metadata_file}")


            # Validate projection files
            projections_path = dataset_dir / "projections"
            if projections_path.is_dir():
                if dataset_schemas.get("projections"):
                    logging.info(f"Validating projection files for {dataset_name} in {projections_path}...")
                    for proj_file in sorted(projections_path.glob("*.json")):
                        validate_file(proj_file, dataset_schemas["projections"], f"Projection file for {dataset_name}")
                else:
                    logging.warning(f"SKIPPED: Projections directory found for '{dataset_name}' but no projection schema defined: {projections_path}")
            elif dataset_schemas.get("projections"): # Schema exists but dir doesn't
                logging.warning(f"SKIPPED: Projections schema defined for '{dataset_name}' but directory not found: {projections_path}")


            # Validate timeseries files
            timeseries_path = dataset_dir / "timeseries"
            if timeseries_path.is_dir():
                if dataset_schemas.get("timeseries"):
                    logging.info(f"Validating timeseries files for {dataset_name} in {timeseries_path}...")
                    for ts_file in sorted(timeseries_path.glob("*.json")):
                        validate_file(ts_file, dataset_schemas["timeseries"], f"Timeseries file for {dataset_name}")
                else:
                    logging.warning(f"SKIPPED: Timeseries directory found for '{dataset_name}' but no timeseries schema defined: {timeseries_path}")
            elif dataset_schemas.get("timeseries"): # Schema exists but dir doesn't
                 logging.warning(f"SKIPPED: Timeseries schema defined for '{dataset_name}' but directory not found: {timeseries_path}")
    
    # --- Summary ---
    logging.info("\n--- Validation Summary ---")
    logging.info(f"{validated_file_count} files checked.")
    if error_count > 0:
        logging.error(f"{error_count} validation error(s) found.")
        sys.exit(1)
    else:
        logging.info("All checked files passed validation.")
        sys.exit(0)

if __name__ == "__main__":
    main()
