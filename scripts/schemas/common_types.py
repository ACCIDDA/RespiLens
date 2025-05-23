from typing import List, Union, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl, confloat, conint, Field, validator
from datetime import datetime
# Try importing Annotated from typing_extensions first, then typing for broader compatibility
try:
    from typing_extensions import Annotated
except ImportError:
    from typing import Annotated

# Basic types
LocationAbbrev = Annotated[str, Field(pattern=r"^[A-Z0-9]{2,3}$")] # US, AL, 01 etc.
DateStr = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")] # YYYY-MM-DD

class LocationBase(BaseModel):
    location: LocationAbbrev
    abbreviation: str # Could be the same as location or different (e.g. US vs US)
    name: str
    population: Optional[conint(ge=0)] = None # Optional population

# Model for quantile predictions (most common in current scripts)
class QuantilePrediction(BaseModel):
    date: DateStr # Changed from target_end_date
    quantiles: List[confloat(ge=0, le=1)]
    values: List[Optional[float]] # Allow for nulls if data can be missing

    @validator('values')
    def check_quantiles_values_length(cls, v, values):
        if 'quantiles' in values and len(v) != len(values['quantiles']):
            raise ValueError('quantiles and values lists must have the same length')
        return v

class ModelPredictions(BaseModel):
    type: Annotated[str, Field(pattern=r"^(quantile|point|pmf|sample)$")] # Extend as needed
    # predictions is a dictionary where keys are horizon strings e.g. "0", "1", "-1"
    # Using str for keys; regex validation for dict keys can be done via a separate validator if essential
    predictions: Dict[str, QuantilePrediction]

    @validator('predictions')
    def check_prediction_keys_are_valid_horizons(cls, v):
        for key in v.keys():
            if not isinstance(key, str) or not key.strip('-').isdigit(): # Simple check for "integer-like strings"
                 raise ValueError(f"Prediction key '{key}' must be an integer-like string (e.g., '-1', '0', '10').")
        return v
