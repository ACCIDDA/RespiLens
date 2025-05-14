"""
This module intends to convert raw CDC json data pulled into a format that is compatible with RespiLens.

Input file format is very specific to how the CDC API structures data. 

Functions:
    respilens_data_converter: Converts CDC data into RespiLens format
    save_data: Saves output of respilens_data_converter locally as .jsons.
    path_to_object: Saves files as objects based on given pathway.
"""


import argparse
import copy
import json
from pathlib import Path


def respilens_data_converter(data: list, metadata: dict) -> dict:
    """
    Converts a list of json-formatted CDC data into a specified structure for RespiLens.
    
    Args:
        data: Raw json CDC data, encased in a list.
        metadata: Corresponding CDC json metadata. 
        
    Returns:
        return_data:
            A dict of RespiLens-friendly json data.
            e.g., 'CA' jurisdiction can be accessed by `return_data[respilens_data_CA]`.
    """
    
    # Establish RespiLens json format
    respilens_data_structure = {
    "metadata": {"location": None, "abbreviation": None, "location_name": None, "population": None},
    "ground_truth": {"dates": [], "values": []},
    "data": {"official": {}}
    }
    
    # Put empty columns into RespiLens data structure
    columns = []
    for column in metadata['columns']:
        columns.append(column['fieldName']) # Using short-form column names
    for column in columns:
        respilens_data_structure['data']['official'][column] = []
        
    # Make a dictionary to consolidate `data` by jurisdiction
    combined_jurisdictions = {}
    for entry in data:
        combined_jurisdictions[entry['jurisdiction']] = {}
        for column in columns:
            combined_jurisdictions[entry['jurisdiction']][column] = []
            
    # Populate combined jurisdiction information with lists of info
    for entry in data:
        for column in entry:
            combined_jurisdictions[entry['jurisdiction']][column].append(entry[column])
    # Remove redundant jurisdiction information
    for jurisdiction in combined_jurisdictions:
        del combined_jurisdictions[jurisdiction]['jurisdiction']
        
    # Populate RespiLens data structures with data from each jurisdiction
    base_string = "respilens_data_"
    return_data = {}
    for jurisdiction in combined_jurisdictions:
        respilens_data = copy.deepcopy(respilens_data_structure) # use the RespiLens structure dict as a basis
        # Fill in location abbrv. and dates (because they are stored separate from column info)
        respilens_data['metadata']['location'] = jurisdiction
        respilens_data['ground_truth']['dates'] = combined_jurisdictions[jurisdiction]['weekendingdate']
        # Fill in column info into 'data' key of RespiLens structure
        for column, values in combined_jurisdictions[jurisdiction].items():
            respilens_data['data']['official'][column].extend(values)
        
        # Remove unwanted keys in 'data' after iteration (these are stored elsewhere in the dict)
        if 'weekendingdate' in respilens_data['data']['official']:
            del respilens_data['data']['official']['weekendingdate']
        if 'jurisdiction' in respilens_data['data']['official']:
            del respilens_data['data']['official']['jurisdiction']
            
        # Add this jurisdiction's dictionary to return_data dict
        return_data[base_string + jurisdiction] = respilens_data
        
    
    return return_data


def save_data(agg_data: dict, output_path: str) -> None:
    """
    Saves aggregated data locally in individual .json files.
    
    Args:
        agg_data: A dictionary containing json data, separated by jurisdiction.
        output_path: Path for data to be stored at.
    """
    
    output_path = Path(output_path) # convert to Path
    # Create directories
    directory = output_path / "respilens_data_cdc"
    directory.mkdir(parents = True, exist_ok = True)
    
    # Save data in individual .json files
    for jurisdiction, json in agg_data.items():
        with open(f"{directory}/{jurisdiction}.json", "w") as file:
            json.dump(json, file, indent = 4)

            
def path_to_object(file_path: str) -> list | dict:
    """
    Takes in file pathway to json data and returns the file as an object.
    
    Args:
        file_path: Pathway to file.
        
    Returns:
        If file is metadata, a dict is returned.
        If file is data, a list is returned.
        
        Both are raw json.
    """
    
    with open(file_path, "r") as file:
        json = json.load(file)
    
    return json
 
    
def main(): 
    """
    Main execution function.
    """
    
    # Create CL options
    parser = argparse.ArgumentParser(description = "Convert data from CDCData downloader to RespiLens-friendly format.")
    parser.add_argument("--cdc-data-path",
                        type = str,
                        help = "Path to CDCData downloader data .json file.")
    parser.add_argument("--cdc-metadata-path",
                        type = str,
                        help = "Path to the CDC Data downloader metadata .json file.")
    parser.add_argument("--output-path",
                        type = str,
                        help = "Pathway to directory to save data to.")
    args = parser.parse_args()
    
    # Use pathways to store json as Python objects 
    data = path_to_object(args.cdc_data_path) # should be a list
    metadata = path_to_object(args.cdc_metadata_path) # should be a dict
    
    # Convert the data to RespiLens format and store in aggregate dictionary
    agg_data = respilens_data_converter(data, metadata)
    
    # Save data locally; files separated by jurisdiction
    save_data(agg_data, args.output_path)


if __name__ == "__main__":
    main()