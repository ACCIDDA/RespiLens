from typing import List, Union, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl, constr, confloat, conint, validator
from datetime import datetime

# Basic types
LocationAbbrev = constr(regex=r"^[A-Z0-9]{2,3}$") # US, AL, 01 etc.
DateStr = constr(regex=r"^\d{4}-\d{2}-\d{2}$") # YYYY-MM-DD

class LocationBase(BaseModel):
    location: LocationAbbrev
    abbreviation: str # Could be the same as location or different (e.g. US vs US)
    name: str
    population: Optional[conint(ge=0)] = None # Optional population

# Model for quantile predictions (most common in current scripts)
class QuantilePrediction(BaseModel):
    target_end_date: DateStr
    quantiles: List[confloat(ge=0, le=1)]
    values: List[Union[float, None]] # Allow for nulls if data can be missing

    @validator('values')
    def check_quantiles_values_length(cls, v, values):
        if 'quantiles' in values and len(v) != len(values['quantiles']):
            raise ValueError('quantiles and values lists must have the same length')
        return v

class ModelPredictions(BaseModel):
    type: constr(regex=r"^(quantile|point|pmf|sample)$") # Extend as needed
    # predictions is a dictionary where keys are horizon strings e.g. "0", "1", "-1"
    predictions: Dict[constr(regex=r"^-?\d+$"), QuantilePrediction] # Defaulting to QuantilePrediction as it's most detailed in spec
