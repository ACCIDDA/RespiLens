from typing import List, Dict, Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime
from .common_types import LocationBase # LocationBase is suitable here

# For the main processed_data/locations.json
class LocationsFile(BaseModel):
    # The file is a list of location objects
    __root__: List[LocationBase]

# For the dataset entries within the global processed_data/metadata.json
class GlobalDatasetEntry(BaseModel):
    fullName: str
    views: List[str] # e.g. ["detailed", "timeseries"]
    prefix: str # e.g. "flu", "rsv", "nhsn" - used for URL params, etc.
    dataPath: str # e.g. "datasets/flusight/projections", "datasets/nhsn/timeseries"
    hasModelSelector: Optional[bool] = None
    hasDateSelector: Optional[bool] = None
    # Other fields from data-format.md Section 3 can be added here if needed by the application
    # For example, defaultModel, defaultView (though defaultView is more of a UI concern)

# For the global processed_data/metadata.json
class GlobalMetadataFile(BaseModel):
    build_timestamp: datetime
    datasets: Dict[str, GlobalDatasetEntry] # Keyed by dataset shortName e.g. "flusight", "rsv_hub", "nhsn"
    demo_mode: Optional[bool] = None
