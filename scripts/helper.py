"""Helper functions for data conversion process."""

import json
from typing import Literal
import numpy as np
import pandas as pd
import requests
import time 
import logging 
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

logger = logging.getLogger(__name__)


def clean_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """Purge dfs of JSON-incompatible `NaN` values."""
    return df.replace({np.nan: None})


def hubverse_df_preprocessor(df: pd.DataFrame, filter_quantiles: bool = True, filter_nowcasts: bool = True) -> pd.DataFrame:
    """
    Do a number of pre-processing tasks that make a hubverse df ready to pass through a processing class.

    Args:
        df: Hubverse data as pd.DataFrame
        filter_quantiles: If True, filter to specific quantile/pmf values only
        filter_nowcasts: If True, filter out all negative horizons (including nowcasts)

    Returns:
        A df with...
            - horizons as ints,
            - horizon NaNs dropped,
            - negative horizons (including nowcasts) optionally filtered out,
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
    # Optionally filter out all negative horizons (including nowcasts)
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


def samples_to_quantiles(df: pd.DataFrame, quantile_levels: tuple = (0.025, 0.25, 0.5, 0.75, 0.975)) -> pd.DataFrame:
    """
    Summarize `output_type` == sample rows into quantile rows, so sample-only submissions can be displayed.

    Hubs without a `horizon` task id (e.g., flu-metrocast 2026-27) get horizons derived as whole weeks
    between `reference_date` and `target_end_date`. Non-sample rows are returned unchanged.

    Args:
        df: Hubverse data as pd.DataFrame
        quantile_levels: Quantile levels to compute from the samples

    Returns:
        The df with sample rows replaced by quantile rows.
    """
    is_sample = df['output_type'] == 'sample'
    if not is_sample.any():
        return df
    samples = df[is_sample].copy()
    missing_horizon = samples['horizon'].isna()
    if missing_horizon.any():
        weeks = (pd.to_datetime(samples.loc[missing_horizon, 'target_end_date'])
                 - pd.to_datetime(samples.loc[missing_horizon, 'reference_date'])).dt.days // 7
        samples.loc[missing_horizon, 'horizon'] = weeks
    task_cols = [col for col in df.columns if col not in ('output_type', 'output_type_id', 'value')]
    quantiles = (
        samples.groupby(task_cols, dropna=False)['value']
        .quantile(list(quantile_levels))
        .rename('value')
        .reset_index()
        .rename(columns={f'level_{len(task_cols)}': 'output_type_id'})
    )
    quantiles['output_type'] = 'quantile'
    return pd.concat([df[~is_sample], quantiles[df.columns]], ignore_index=True)


def samples_to_peaks(
    df: pd.DataFrame,
    target_data: pd.DataFrame | None = None,
    quantile_levels: tuple = (0.025, 0.25, 0.5, 0.75, 0.975),
    short_term_max_horizon: int = 3,
) -> dict:
    """
    Peak forecasts of long-term sample trajectories: the size and week of each sample season's
    highest value, summarized over the samples (as the narrative does). This is the forecast of the
    peak itself, which runs higher than the high point of the median line.

    Each sample season is the whole season: the weeks observed before the forecast (from
    `target_data`, as available on the forecast date) followed by the sample's own weeks. A season
    that already peaked higher than a sample ever goes peaks in its observed week.

    Only trajectories reaching past `short_term_max_horizon` weeks count (a 4-week forecast has no
    peak to speak of), and only samples covering every week of their trajectory.

    Returns:
        {location: {reference_date: {f"peak {target}": {model: quantiles of the peak size},
                                     f"peak week {target}": {model: pmf over peak weeks}}}},
        in the shape of FluSight's `peaks` key.
    """
    samples = df[(df['output_type'] == 'sample') & df['value'].notna()]
    if samples.empty:
        return {}
    keys = ['model_id', 'reference_date', 'location', 'target']
    samples = samples[keys + ['output_type_id', 'target_end_date', 'value']].copy()
    for col in ('reference_date', 'target_end_date'):
        samples[col] = pd.to_datetime(samples[col]).dt.strftime('%Y-%m-%d')
    weeks_ahead = (pd.to_datetime(samples['target_end_date']) - pd.to_datetime(samples['reference_date'])).dt.days // 7
    samples = samples[weeks_ahead.groupby([samples[k] for k in keys]).transform('max') > short_term_max_horizon]
    if samples.empty:
        return {}
    # Whole sample seasons only
    per_sample = samples.groupby(keys + ['output_type_id'])['target_end_date'].transform('size')
    samples = samples[per_sample == per_sample.groupby([samples[k] for k in keys]).transform('max')]
    # Each sample's first highest week
    samples = samples.sort_values('target_end_date')
    peaks_df = samples.loc[samples.groupby(keys + ['output_type_id'], sort=False)['value'].idxmax()]
    if target_data is not None and not target_data.empty:
        peaks_df = _with_observed_peaks(peaks_df, samples, target_data, keys)

    peaks: dict = {}
    for (model, reference_date, location, target), group in peaks_df.groupby(keys):
        by_date = peaks.setdefault(str(location), {}).setdefault(reference_date, {})
        sizes = group['value']
        by_date.setdefault(f"peak {target}", {})[str(model)] = {
            "type": "quantile",
            "predictions": {
                "quantiles": list(quantile_levels),
                "values": [round(float(sizes.quantile(level)), 4) for level in quantile_levels],
            },
        }
        weeks = group['target_end_date'].value_counts(normalize=True).sort_index()
        by_date.setdefault(f"peak week {target}", {})[str(model)] = {
            "type": "pmf",
            "predictions": {
                "peak week": list(weeks.index),
                "probabilities": [round(float(p), 4) for p in weeks.values],
            },
        }
    return peaks


def _observed_season_peaks(target_data: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    For each forecast (reference_date, location, target, first_week): the season's highest observed
    week before `first_week`, from the data as of the forecast date (the latest version if none is
    that old). Seasons start August 1. Columns: reference_date, location, target, obs_week, obs_value.
    """
    truth = target_data[['location', 'target', 'target_end_date', 'observation']].copy()
    truth['as_of'] = pd.to_datetime(target_data['as_of']) if 'as_of' in target_data else pd.NaT
    truth['target_end_date'] = pd.to_datetime(truth['target_end_date']).dt.strftime('%Y-%m-%d')
    truth['observation'] = pd.to_numeric(truth['observation'], errors='coerce')
    truth = truth[truth['observation'].notna()]
    rows = []
    for (location, target), series in truth.groupby(['location', 'target']):
        mine = forecasts[(forecasts['location'] == location) & (forecasts['target'] == target)]
        for reference_date, first_week in mine[['reference_date', 'first_week']].drop_duplicates().itertuples(index=False):
            ref = pd.Timestamp(reference_date)
            versions = series[series['as_of'] <= ref] if series['as_of'].notna().any() else series
            if versions.empty:
                versions = series
            if versions['as_of'].notna().any():
                versions = versions[versions['as_of'] == versions['as_of'].max()]
            season_start = f"{ref.year if ref.month >= 8 else ref.year - 1}-08-01"
            seen = versions[(versions['target_end_date'] >= season_start)
                            & (versions['target_end_date'] < first_week)].sort_values('target_end_date')
            if seen.empty:
                continue
            top = seen.loc[seen['observation'].idxmax()]
            rows.append((reference_date, location, target, top['target_end_date'], float(top['observation'])))
    return pd.DataFrame(rows, columns=['reference_date', 'location', 'target', 'obs_week', 'obs_value'])


def _with_observed_peaks(peaks_df: pd.DataFrame, samples: pd.DataFrame, target_data: pd.DataFrame,
                         keys: list) -> pd.DataFrame:
    """Each sample's peak over the whole season: its own, or the observed one if that is as high."""
    forecasts = (samples.groupby(keys)['target_end_date'].min().rename('first_week').reset_index()
                 [['reference_date', 'location', 'target', 'first_week']].drop_duplicates())
    observed = _observed_season_peaks(target_data, forecasts)
    if observed.empty:
        return peaks_df
    merged = peaks_df.merge(observed, on=['reference_date', 'location', 'target'], how='left')
    # Ties go to the observed week, the earlier one
    use_observed = merged['obs_value'].notna() & (merged['obs_value'] >= merged['value'])
    merged.loc[use_observed, 'value'] = merged.loc[use_observed, 'obs_value']
    merged.loc[use_observed, 'target_end_date'] = merged.loc[use_observed, 'obs_week']
    return merged.drop(columns=['obs_week', 'obs_value'])


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
        "california": "06",
        "colorado": "08",
        "georgia": "13",
        "indiana": "18",
        "maine": "23",
        "maryland": "24",
        "massachusetts": "25",
        "minnesota": "27",
        "new-york": "36",
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
    # Only HSA ids are unique across states; state-specific region ids (ut_region_id, nc_flu_region_id,
    # ca_region_id) restart at 1 or are empty, so those locations are keyed by their slug instead
    if (value_needed == 'original_location_code') and ('location_type' in current_df.columns) \
            and (current_df['location_type'].iloc[0] != 'hsa_nci_id'):
        return location
    else:
        return str(current_df[value_needed].iloc[0])
    
def retrieve_data_from_endpoint_aslist(data_url: str, query_params: dict | None = None) -> list[dict]:
    """Downloads CDC data from API endpoint with pagination and retries.

    `query_params` are extra SoQL parameters (e.g. `$select`, `$where`) sent with every page.
    """
        
    session = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=1,
                    status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    all_data = []
    offset = 0
    batch_size = 50000 # Socrata's documented max page size
    # paging with $offset needs a stable order, otherwise rows can be skipped or repeated
    query_params = {"$order": ":id", **(query_params or {})}
    while True:
        params = {**query_params, "$limit": batch_size, "$offset": offset}
        try:
            # Use the configured session to make the request
            data_response = session.get(data_url, params=params, timeout=300)
            data_response.raise_for_status()
            batch_data = data_response.json()
            if not batch_data:
                break
            all_data.extend(batch_data)
            if len(batch_data) < batch_size:
                break
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
