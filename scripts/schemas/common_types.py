import math # Add for math.isclose
from typing import List, Union, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl, confloat, conint, Field, validator, root_validator
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
    predictions: Dict[str, Union[QuantilePrediction, PMFPrediction]] # Updated type hint

    @validator('predictions') # This existing validator checks keys like "-1", "0"
    def check_prediction_keys_are_valid_horizons(cls, v):
        for key in v.keys():
            if not isinstance(key, str) or not key.strip('-').isdigit():
                 raise ValueError(f"Prediction key '{key}' must be an integer-like string (e.g., '-1', '0', '10').")
        return v

    @root_validator(pre=False, skip_on_failure=True) # post-validation, skip if other field validations failed
    def check_predictions_match_type(cls, values):
        # 'values' here is a dict of the model's fields after individual field validation
        model_type = values.get('type')
        predictions_dict = values.get('predictions')

        if not model_type or not predictions_dict:
            # This should not happen if 'type' and 'predictions' are required fields
            # or if skip_on_failure=True handles it.
            # Can raise an error or return values if other validators should catch missing type/predictions.
            return values # Or raise error if type/predictions are mandatory and somehow missing here

        for horizon_key, prediction_obj in predictions_dict.items():
            if model_type == "quantile":
                if not isinstance(prediction_obj, QuantilePrediction):
                    raise ValueError(
                        f"For model type 'quantile', prediction for horizon '{horizon_key}' "
                        f"must be a QuantilePrediction object, got {type(prediction_obj).__name__}."
                    )
            elif model_type == "pmf":
                if not isinstance(prediction_obj, PMFPrediction):
                    raise ValueError(
                        f"For model type 'pmf', prediction for horizon '{horizon_key}' "
                        f"must be a PMFPrediction object, got {type(prediction_obj).__name__}."
                    )
            # Add elif for other types like 'point', 'sample' when their models are defined
            # else:
                # This case should ideally not be reached if 'type' field is validated against allowed enum/literal
                # raise ValueError(f"Unknown model type '{model_type}' encountered in validator.")

        return values

# Model for PMF (Probability Mass Function) predictions
class PMFPrediction(BaseModel):
    date: DateStr
    categories: List[str]
    probabilities: List[confloat(ge=0, le=1)]

    @validator('probabilities')
    def check_probabilities_sum(cls, v):
        if not math.isclose(sum(v), 1.0, abs_tol=0.001): # Allow for small floating point inaccuracies
            raise ValueError('Probabilities must sum to 1.0')
        return v

    @validator('probabilities')
    def check_categories_probabilities_length(cls, v, values):
        # This validator style is for Pydantic V1 where 'values' contains already validated fields
        if 'categories' in values and len(v) != len(values['categories']):
            raise ValueError('categories and probabilities lists must have the same length')
        return v
