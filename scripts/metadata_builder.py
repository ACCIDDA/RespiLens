"""
Script for building RespiLens-style metadata.

Functions:
    metadata_builder: Builds RespiLens metadata from user input; min requirement is dataset short name.
"""

from datetime import date


def metadata_builder(shortName: str = "", fullName: str = "", defaultView: str = "", datasetType: str = "") -> dict:
    """
    Builds RespiLens metadata from user input.

    Args:
        shortName: Short or abbreviated name of the dataset data was pulled from.
        fullName: Full name of the dataset data was pulled from.
        defaultView: Which view the data belongs in.
        datasetType: The type of data. 
    """

    metadata_struct = { 
        "shortName": f"{shortName}",
        "fullName": f"{fullName}",
        "defaultView": f"{defaultView}",
        "lastUpdated": date.today().strftime("%Y-%m-%d"),
        "datasetType": f"{datasetType}"
    }
    return metadata_struct
