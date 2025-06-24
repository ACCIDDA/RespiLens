"""
Script for saving RespiLens data locally.

Functions:
    save_data: Saves data at specified output path; titled <locationabbrv>.json.
    save_metadata: Saves metadata at specified output path; titled metadata.json 
"""

import json
import os
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
with open(SCRIPT_DIR / "respilens-data.schema.json", "r") as f:
    RESPILENS_DATA_SCHEMA = json.load(f)


def save_data(data: dict, output_path: str | Path) -> None:
    """
    Function to save RespiLens data to a local directory.

    Args:
        data: RespiLens-formatted json data to be saved.
        output_path: Path to directory where data will be saved.
    """
    # Ensure directory exists
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)
    # Validate data with RespiLens jsonschema
    try:
        validate(instance=data, schema=RESPILENS_DATA_SCHEMA)
    except ValidationError as e:
        raise ValueError("Data does not match RespiLens jsonschema.") from e

    # Save data 
    output_file = os.path.join(output_path, f"{data["metadata"]["location"]}.json") 
    with open(output_file, "w") as data_json_file:
        json.dump(data, data_json_file, indent=4)
    

def save_metadata(metadata: dict, output_path: str | Path) -> None:
    """
    Function to save RespiLens metadata to a local directory.

    Convention is 1 metadata file per dataset.

    Args:
        metadata: Metadata to be saved.
        output_path: Path to directory where metadata will be saved. 
    """

    # Ensure directory exists
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Save metadata
    with open(f"{output_path}/metadata.json", "w") as metadata_json_file:
        json.dump(metadata, metadata_json_file, indent = 4) 
