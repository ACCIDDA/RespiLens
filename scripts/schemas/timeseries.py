from typing import List, Dict, Optional, Union
from pydantic import BaseModel, validator, root_validator, constr
from .common_types import DateStr, LocationAbbrev, LocationBase

class SeriesMetadata(LocationBase): # Reusing LocationBase for common fields
    dataset: str # e.g., "nhsn"
    series_type: constr(regex=r"^(official|preliminary)$")

class SeriesData(BaseModel):
    dates: List[DateStr]
    # Columns are Dict[column_name, List_of_values]. Values can be float, int, or None.
    columns: Dict[str, List[Optional[Union[float, int]]]]

    @validator('columns')
    def check_column_lengths_match_dates(cls, v, values):
        if 'dates' in values:
            dates_len = len(values['dates'])
            for col_name, col_values in v.items():
                if len(col_values) != dates_len:
                    raise ValueError(
                        f"Column '{col_name}' has length {len(col_values)}, "
                        f"but dates list has length {dates_len}."
                    )
        return v

class SingleTimeseries(BaseModel):
    metadata: SeriesMetadata
    series: SeriesData

# For NHSN files that contain both "official" and "preliminary" keys
class NHSNLocationTimeseriesFile(BaseModel):
    official: Optional[SingleTimeseries] = None
    preliminary: Optional[SingleTimeseries] = None

    @root_validator
    def check_at_least_one_series(cls, values):
        if not values.get('official') and not values.get('preliminary'):
            raise ValueError("At least one of 'official' or 'preliminary' series must be present.")
        return values
