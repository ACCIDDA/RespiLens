from typing import Dict, List, Optional
from pydantic import BaseModel
from .common_types import DateStr, ModelPredictions, LocationBase

# Basic building blocks for forecast data
ModelNameStr = str
ModelData = Dict[ModelNameStr, ModelPredictions]

TargetNameStr = str
TargetData = Dict[TargetNameStr, ModelData]

# Forecast structure for FluSight-like data (no age groups)
# e.g., "2025-02-01": { "wk inc flu hosp": { "modelA": { ... } } }
Forecasts = Dict[DateStr, TargetData]

# Forecast structure for RSV-like data (includes age groups)
AgeGroupStr = str
# e.g., "0-4": { "inc hosp": { "modelA": { ... } } }
RSVTargetDataByAgeGroup = Dict[AgeGroupStr, TargetData]
# e.g., "2025-02-01": { "0-4": { "inc hosp": { "modelA": { ... } } } }
RSVForecasts = Dict[DateStr, RSVTargetDataByAgeGroup]


# Common metadata structure for individual projection files
# This metadata is specific to the location file itself
class FileMetadata(LocationBase): # Inherits location, abbreviation, name, population
    dataset: str # e.g., "flusight", "rsv_hub"
    # location field is inherited from LocationBase, which includes location, abbreviation, name, population

# Top-level model for FluSight location projection files
class FluSightLocationProjectionsFile(BaseModel):
    metadata: FileMetadata
    forecasts: Forecasts
    # No ground_truth or available_models as per V2 projection spec (data-format.md Section 7)

# Top-level model for RSV location projection files
class RSVLocationProjectionsFile(BaseModel):
    metadata: FileMetadata
    forecasts: RSVForecasts
    # No ground_truth or available_models as per V2 projection spec (data-format.md Section 7)
