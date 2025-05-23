from typing import List, Optional
from pydantic import BaseModel, validator, HttpUrl
from datetime import datetime
from .common_types import LocationBase # Assuming common_types.py is in the same directory

class BaseDatasetMetadata(BaseModel):
    shortName: str
    fullName: str
    defaultView: Optional[str] = None
    targets: List[str]
    quantile_levels: Optional[List[float]] = None # Optional as not all datasets might have it
    last_updated: Optional[datetime] = None
    models: Optional[List[str]] = None # List of model names
    # Kept locations for now per previous discussion, can be removed if global locations.json is sole source
    locations: Optional[List[LocationBase]] = None
    demo_mode: Optional[bool] = None

class FluSightDatasetMetadata(BaseDatasetMetadata):
    shortName: str = "flusight"
    # Example targets, adjust if script uses different ones
    targets: List[str] = ["wk inc flu hosp", "wk flu hosp rate change", "peak week", "peak inc"]
    # Example quantiles, adjust from script if different
    quantile_levels: List[float] = [
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
    ]

class RSVHubDatasetMetadata(BaseDatasetMetadata):
    shortName: str = "rsv_hub"
    targets: List[str] = ["inc hosp"] # Main target for RSV
    age_groups: Optional[List[str]] = None
    quantile_levels: List[float] = [ # Example
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
    ]

class NHSNDatasetMetadata(BaseModel): # NHSN is simpler as per data-format.md
    shortName: str = "nhsn"
    fullName: str = "National Healthcare Safety Network Data"
    description: Optional[str] = None
    last_updated: Optional[datetime] = None
    # No targets/quantiles specified for NHSN dataset-level metadata in data-format.md
    # It's primarily timeseries columns.
