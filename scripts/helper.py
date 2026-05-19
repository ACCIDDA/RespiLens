"""Helper functions for data conversion process."""

import json 
import jsonschema
from typing import Literal
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import time 
import logging 
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Import schema
_current_dir = Path(__file__).parent
projections_schema_path = _current_dir / 'schemas' / 'RespiLens_projections.schema.json'
timeseries_schema_path = _current_dir / 'schemas' / 'RespiLens_timeseries.schema.json'
with open(projections_schema_path, 'r') as f:
    projections_schema = json.load(f)
with open(timeseries_schema_path, 'r') as f:
    timeseries_schema = json.load(f)


def clean_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """Purge dfs of JSON-incompatible `NaN` values."""
    return df.replace({np.nan: None})


def hubverse_df_preprocessor(df: pd.DataFrame, filter_quantiles: bool = True, filter_nowcasts: bool = True) -> pd.DataFrame:
    """
    Do a number of pre-processing tasks that make a hubverse df ready to pass through a processing class.

    Args:
        df: Hubverse data as pd.DataFrame
        filter_quantiles: If True, filter to specific quantile/pmf values only
        filter_nowcasts: If True, filter out horizon -1 (nowcasts)

    Returns:
        A df with...
            - horizons as ints,
            - horizon NaNs dropped,
            - horizon -1 (nowcasts) optionally filtered out,
            - only some `output_type_id` values kept (if filter_quantiles=True),
            - all `output_type` == sample removed.
    """
    df = df.copy()
    # Set horizon for flu 'peak' targets = 50 (placeholder so it doesn't get filtered out)
    peak_targets = {'peak inc flu hosp', 'peak week inc flu hosp'}
    if 'target' in df.columns:
        df['target'] = df['target'].astype(str)
        is_peak_target = df['target'].isin(peak_targets)
        df.loc[is_peak_target, 'horizon'] = 50
    # Drop NaN values in horizon column
    df = df.dropna(subset=['horizon'])
    # Ensure horizons are ints (not floats)
    df['horizon'] = df['horizon'].astype(int)
    # Optionally filter out horizon -1 (nowcasts)
    if filter_nowcasts:
        df = df[df['horizon'] >= 0]
    # Get rid of all output_type == 'sample'
    df = df[df['output_type'] != 'sample']
    if filter_quantiles:
        # Filter `output_type_id` values
        categorical_ids = ['decrease', 'increase', 'large_decrease', 'large_increase', 'stable'] 
        numeric_ids = [0.025, 0.25, 0.5, 0.75, 0.975]
        df['output_type_id'] = df['output_type_id'].astype(object)
        # Also valid, date-like values where target is 'peak' something
        numeric_output_ids = pd.to_numeric(df['output_type_id'], errors='coerce')
        is_categorical_id = df['output_type_id'].isin(categorical_ids)
        is_numeric_id = numeric_output_ids.isin(numeric_ids)
        date_output_ids = pd.to_datetime(df['output_type_id'], errors='coerce')
        is_convertible_to_date = date_output_ids.notna()
        is_peak_week_target = df['target'].astype(str).str.contains('peak week inc flu hosp', na=False)
        is_valid_peak_date = is_convertible_to_date & is_peak_week_target
        df = df[is_categorical_id | is_numeric_id | is_valid_peak_date]
        quantile_mask = df['output_type'] == 'quantile'
        df.loc[quantile_mask, 'output_type_id'] = df.loc[quantile_mask, 'output_type_id'].astype(float)

    return df


def get_location_info(
        location_data: pd.DataFrame, 
        location: str, 
        value_needed: Literal['abbreviation', 'location_name', 'population', 'original_location_code'],
) -> str:
    """
    Get a variety of location metadata information given the FIPS code of a location.

    Args:
        location_data: The df of location metadata
        location: FIPS code for location for which info will be retrieved ('US' for US)
        value_needed: Which piece of info to retrieve (one of 'abbreviation', 'location_name', 'population', 'original_location_code')

    Returns:
        The value requested (as a str)

    Raises:
        ValueError: 
            If the location FIPS code provided via `location` param is not in the location metadata
    """
    # The FIPS codes are unnecessary for respi needs (metrocast non-state locs use HSA id)
    # But Respi DOES require that they be unique, and the default code value for metrocast states is 'All'
    # So here we choose to use state FIPS codes instead of 'All' to make them keys
    metrocast_states_to_fips = {
        "colorado": "08",
        "georgia": "13",
        "indiana": "18",
        "maine": "23",
        "maryland": "24",
        "massachusetts": "25",
        "minnesota": "27",
        "north-carolina": "37",
        "oregon": "41",
        "south-carolina": "45",
        "texas": "OVERLAP-WITH-frederick_md", # putting this b/c the Texas FIPS is the same as Frederick, MD HSAid (48)
        "utah": "49",
        "virginia": "51"
    }
    current_df = location_data[location_data['location'] == location]
    if current_df.empty:
        raise ValueError(f"Could not find location {location} in location data.")
    if value_needed == 'population':
        return int(current_df[value_needed].iloc[0])
    if (value_needed == 'original_location_code') and (location in metrocast_states_to_fips.keys()):
        try: return metrocast_states_to_fips[location]
        except KeyError:
            raise KeyError(f"Flu MetroCast has added a new state {location}. Update `metrocast_states_to_fips` to include this state.")
    else:
        return str(current_df[value_needed].iloc[0])
    
def retrieve_data_from_endpoint_aslist(data_url: str) -> list[dict]:
    """Downloads CDC data from API endpoint with pagination and retries."""
        
    session = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=1,
                    status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    all_data = []
    offset = 0
    batch_size = 1000
    while True:
        params = {"$limit": batch_size, "$offset": offset}
        try:
            # Use the configured session to make the request
            data_response = session.get(data_url, params=params, timeout=30)
            data_response.raise_for_status()
            batch_data = data_response.json()
            if not batch_data:
                break
            all_data.extend(batch_data)
            offset += batch_size
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
    return all_data
    

def save_json_file(
        pathogen: Literal['flusight', 'flu', 'flusightforecasthub', 'rsv','covid','covid19','rsvforecasthub','covid19forecasthub','nhsn', 'nssp', 'flumetrocast', 'flumetrocasthub'],
        output_path: str,
        output_filename: str,
        file_contents: dict,
        overwrite: bool
) -> None:
    """
    Save an already-validated JSON to output_path/pathogen-ext/file_name.json.

    Args:
        pathogen: Type of data in JSON payload (canonical slug or legacy alias)
        output_path: Path to top-level saving directory
        output_filename: Full name of file to be saved
        file_contents: Contents of file to be saved

    Raises:
        FileExistsError: If file already exists at the full output path and overwrite is set to False.
    """

    # This single dictionary maps every possible input to the desired output directory.
    output_dir_map = {
        'flu': 'flusight',
        'flusight': 'flusight',
        'flusightforecasthub': 'flusight',
        'rsv': 'rsvforecasthub',
        'rsvforecasthub': 'rsvforecasthub',
        'covid': 'covid19forecasthub',
        'covid19': 'covid19forecasthub',
        'covid19forecasthub': 'covid19forecasthub',
        'nhsn': 'nhsn',
        'nssp': 'nssp',
        'flumetrocast': 'flumetrocast',
        'flumetrocashtub': 'flumetrocast',
    }

    if pathogen not in output_dir_map:
        raise ValueError(f"Invalid pathogen ('{pathogen}') provided; must be one of {list(output_dir_map.keys())}")

    # Get the single, correct directory name
    target_name = output_dir_map[pathogen]
    
    # Create the full path and save the file (no loop needed)
    target_dir = Path(output_path) / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / output_filename
    
    if (not overwrite) and file_path.exists():
        raise FileExistsError(
            f"Error saving {output_filename}; file already found at {file_path}."
            "Remove or move file and try again."
        )
    
    with open(file_path, 'w') as of:
        json.dump(file_contents, of, indent=4)



def validate_respilens_json(json_contents: dict, type: str = Literal['projections', 'timeseries']) -> bool | str:
    """
    Validate JSON output with RespiLens schema

    Args:
        json_contents: Contents of json file, stored as python dict
        type: Type of RespiLens data (either projections or timeseries)

    Returns:
        True if validation is successful, str of error message if unsuccessful.

    Raise:
        ValidationError: When json contents do not match jsonschema
    """
    if type == 'projections':
        try:
            jsonschema.validate(instance=json_contents, schema=projections_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            return str(e)
    elif type == 'timeseries':
        try:
            jsonschema.validate(instance=json_contents, schema=timeseries_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            return str(e)
    else:
        raise ValueError(f"`type` parameter must be one of 'projections' or 'timeseries'. Received {type}")


NHSN_COLUMN_MASKS = {
    "RAW_PATIENT_COUNTS": [
        'jurisdiction',
        'weekendingdate',
        'totalconfc19newadmadult', 
        'totalconfflunewadmadult',
        'totalconfrsvnewadmadult',
        'totalconfc19newadm',
        'totalconfc19icupats',
        'totalconffluicupats',
        'totalconfrsvicupats',
        'totalconfflunewadm',
        'totalconfc19hosppats',
        'totalconffluhosppats',
        'totalconfrsvhosppats',
        'totalconfc19newadmped',
        'totalconfflunewadmped',
        'totalconfrsvnewadmped',
        'totalconfrsvnewadm',
        'numconfc19newadmadult18to49',
        'numconfc19newadmadult50to64',
        'numconfc19newadmadult65to74',
        'numconfc19newadmadult75plus',
        'numconfc19icupatsadult',
        'numconffluicupatsadult',
        'numconfrsvicupatsadult',
        'numconfflunewadmadult18to49',
        'numconfflunewadmadult50to64',
        'numconfflunewadmadult65to74',
        'numconfflunewadmadult75plus',
        'numconfc19hosppatsadult',
        'numconffluhosppatsadult',
        'numconfrsvhosppatsadult',
        'numconfrsvnewadmadult18to49',
        'numconfrsvnewadmadult50to64',
        'numconfrsvnewadmadult65to74',
        'numconfrsvnewadmadult75plus',
        'numconfc19newadmunk',
        'numconfflunewadmunk',
        'numconffluhosppatsped',
        'numconfc19newadmped0to4',
        'numconfc19newadmped5to17',
        'numconfc19icupatsped',
        'numconffluicupatsped',
        'numconfrsvicupatsped',
        'numconfflunewadmped0to4',
        'numconfflunewadmped5to17',
        'numconfc19hosppatsped',
        'numconfrsvhosppatsped',
        'numconfrsvnewadmped0to4',
        'numconfrsvnewadmped5to17',
        'numconfrsvnewadmunk'
    ],

    "HOSPITAL_ADMISSION_RATES": [
        'jurisdiction',
        'weekendingdate',
        'totalconfc19newadmadultper100k',
        'totalconfflunewadmadultper100k',
        'totalconfrsvnewadmadultper100k',
        'totalconfc19newadmpedper100k',
        'totalconfflunewadmpedper100k',
        'totalconfrsvnewadmpedper100k',
        'totalconfc19newadmper100k',
        'totalconfflunewadmper100k',
        'totalconfrsvnewadmper100k',
        'numconfc19newadmadult18to49per100k',
        'numconfc19newadmadult50to64per100k',
        'numconfc19newadmadult65to74per100k',
        'numconfc19newadmadult75plusper100k',
        'numconfflunewadmadult18to49per100k',
        'numconfflunewadmadult50to64per100k',
        'numconfflunewadmadult65to74per100k',
        'numconfflunewadmadult75plusper100k',
        'numconfrsvnewadmadult18to49per100k',
        'numconfrsvnewadmadult50to64per100k',
        'numconfrsvnewadmadult65to74per100k',
        'numconfrsvnewadmadult75plusper100k',
        'numconfc19newadmped0to4per100k',
        'numconfc19newadmped5to17per100k',
        'numconfflunewadmped0to4per100k',
        'numconfflunewadmped5to17per100k',
        'numconfrsvnewadmped0to4per100k',
        'numconfrsvnewadmped5to17per100k'
    ],

    "HOSPITAL_ADMISSION_PERCENTS": [
        'jurisdiction',
        'weekendingdate',
        'pctconfc19newadmadult',
        'pctconfflunewadmadult',
        'pctconfrsvnewadmadult',
        'pctconfc19newadmped',
        'pctconfflunewadmped',
        'pctconfrsvnewadmped'
    ],

    "RAW_BED_CAPACITY": [
        'jurisdiction',
        'weekendingdate',
        'numicubedsadult',
        'numicubedsoccadult',
        'numinptbedsadult',
        'numinptbedsoccadult',
        'numicubeds',
        'numicubedsocc',
        'numinptbeds',
        'numinptbedsocc',
        'numicubedsped',
        'numicubedsoccped',
        'numinptbedsoccped',
        'numinptbedsped'
    ],

    "CAPACITY_PERCENTS": [
        'jurisdiction',
        'weekendingdate',
        'pcticubedsocc',
        'pctconfc19icubeds',
        'pctconffluicubeds',
        'pctconfrsvicubeds',
        'pctinptbedsocc',
        'pctconfc19inptbeds',
        'pctconffluinptbeds',
        'pctconfrsvinptbeds'
    ],

    "ABSOLUTE_PERCENT_CHANGE": [
        'jurisdiction',
        'weekendingdate',
        'totalconfflunewadmpercho',
        'totalconfc19newadmadultp_1',
        'totalconfflunewadmadultp_1',
        'totalconfrsvnewadmadultp_1',
        'totalconfc19newadmpercho',
        'totalconfc19icupatsperch',
        'totalconffluicupatsperch',
        'totalconfrsvicupatsperch',
        'numicubedsoccperchosprepabschg',
        'numicubedsperchosprepabschg',
        'numinptbedsoccperchospre',
        'numinptbedsperchosprepabschg',
        'totalconfc19newadmpedper_1',
        'totalconfflunewadmpedper_1',
        'totalconfrsvnewadmpedper_1',
        'pctconfc19icubedsperchos',
        'pctconffluicubedsperchos',
        'pctconfrsvicubedsperchos',
        'pcticubedsoccperchosprepabschg',
        'pctconfc19inptbedspercho',
        'pctconffluinptbedspercho',
        'pctconfrsvinptbedspercho',
        'pctinptbedsoccperchospre',
        'totalconfrsvnewadmpercho',
        'totalconfc19hosppatsperc_1',
        'totalconfrsvhosppatsperc_1',
        'totalconffluhosppatsperc_1'
    ]
}

STATENAME_TO_ABBREVIATION_MAP = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'United States': 'US',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
 }

STATEABBREVIATION_TO_FIPS_MAP = {
    'US': 'US',
    'AL': '01',
    'AK': '02',
    'AZ': '04',
    'AR': '05',
    'CA': '06',
    'CO': '08',
    'CT': '09',
    'DE': '10',
    'DC': '11',
    'FL': '12',
    'GA': '13',
    'HI': '15',
    'ID': '16',
    'IL': '17',
    'IN': '18',
    'IA': '19',
    'KS': '20',
    'KY': '21',
    'LA': '22',
    'ME': '23',
    'MD': '24',
    'MA': '25',
    'MI': '26',
    'MN': '27',
    'MS': '28',
    'MO': '29',
    'MT': '30',
    'NE': '31',
    'NV': '32',
    'NH': '33',
    'NJ': '34',
    'NM': '35',
    'NY': '36',
    'NC': '37',
    'ND': '38',
    'OH': '39',
    'OK': '40',
    'OR': '41',
    'PA': '42',
    'RI': '44',
    'SC': '45',
    'SD': '46',
    'TN': '47',
    'TX': '48',
    'UT': '49',
    'VT': '50',
    'VA': '51',
    'WA': '53',
    'WV': '54',
    'WI': '55',
    'WY': '56',
    'PR': '72'
}
