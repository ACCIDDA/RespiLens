# RespiLens Scripts ![respilens-logo](https://github.com/user-attachments/assets/f4b54c2a-9d27-4453-9a85-72b1b2f965a2)

In this `README.md`, you will find brief guides for primary [RespiLens](https://www.RespiLens.com) scripts. Jump to the script you're looking for below:

* [external_to_projections](#external_to_projections)
* [process_RespiLens_data](#process_RespiLens_data)
* [hub_dataset_processor](#hub_dataset_processor)
* [nhsn_data_processor](#nhsn_data_processor)
* [helper](#helper)
* [/processing](#Processing)


## external_to_projections

#### Overview

`external_to_projections.py` is the command-line entry point for converting user Hubverse exports into RespiLens projections JSON files. It pulls raw forecast, target, and location metadata files; validates the inputs; runs the appropriate processor (`processors.flusight`, `processors.rsv_forecast_hub`, or `processors.covid19_forecast_hub`); validates the JSON outputs; and writes a single JSON file per location (plus metadata) using `helper.save_json_file`.

Internally it relies on `external_data.py` to load and validate the incoming files, and on `hub_dataset_processor.HubDataProcessorBase` for the shared processing workflow. Intermediate DataFrames are retained on each processor instance under the `intermediate_dataframes` attribute for optional downstream use.

#### Typical Workflow

1. Gather the three Hubverse exports for your pathogen:
   * `--data-path`: Forecast CSV containing quantiles/pmf predictions.
   * `--target-data-path`: Ground-truth time series (CSV or Parquet, depending on the hub).
   * `--locations-data-path`: Location metadata CSV.
2. Pick the pathogen (`flu`, `rsvforecasthub`, or `covid19forecasthub`) so the loader can enforce pathogen-specific column requirements.
3. Run the CLI pointing at the directory where processed JSON should be written.
4. Each payload is checked against `schemas/RespiLens_projections.schema.json` before being saved via `helper.save_json_file`.
5. Each processor exposes `processor.intermediate_dataframes` if you need to inspect the cleaned DataFrames in downstream tooling.

#### Running `external_to_projections.py`

You can run `external_to_projections.py` with the following options:

| Option | Description | Value Type | Required? | Default Value |
| :--- | :--- | :--- | :--- | :--- |
| `--output-path` | Directory where JSON files will be saved. | String | Yes | *N/A* |
| `--pathogen` | Pathogen to process (`flu`, `rsvforecasthub`, or `covid19forecasthub`). Determines which processor is used. | String | Yes | *N/A* |
| `--data-path` | Absolute path to Hubverse forecast data in CSV format. | String | Yes | *N/A* |
| `--target-data-path` | Absolute path to Hubverse target data (CSV or Parquet). | String | Yes | *N/A* |
| `--locations-data-path` | Absolute path to the location metadata CSV. | String | Yes | *N/A* |
| `--overwrite` | Flag that allows overwriting existing output files. | Flag (Boolean) | No | `False` |
| `--log-level` | Console logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | String | No | `INFO` |

Example command:
```
python scripts/external_to_projections.py \
    --output-path ./app/public/processed_data \
    --pathogen flu \
    --data-path /absolute/path/to/FluSight-forecast-hub/hub_data.csv \
    --target-data-path /absolute/path/to/FluSight-forecast-hub/target-data/time-series.csv \
    --locations-data-path /absolute/path/to/FluSight-forecast-hub/auxiliary-data/locations.csv \
    --overwrite
```

To process other hubs, adjust the pathogen flag and the file paths:

```bash
# RSV Forecast Hub
python scripts/external_to_projections.py \
    --output-path ./app/public/processed_data \
    --pathogen rsvforecasthub \
    --data-path /absolute/path/to/rsv-forecast-hub/hub_data.csv \
    --target-data-path /absolute/path/to/rsv-forecast-hub/target-data/time-series.parquet \
    --locations-data-path /absolute/path/to/rsv-forecast-hub/auxiliary-data/locations.csv \
    --overwrite

# COVID-19 Forecast Hub
python scripts/external_to_projections.py \
    --output-path ./app/public/processed_data \
    --pathogen covid19forecasthub \
    --data-path /absolute/path/to/covid19-forecast-hub/hub_data.csv \
    --target-data-path /absolute/path/to/covid19-forecast-hub/target-data/time-series.parquet \
    --locations-data-path /absolute/path/to/covid19-forecast-hub/auxiliary-data/locations.csv \
    --overwrite
```

The CLI writes each pathogen into its own subdirectory under `--output-path` (`flusight`, `rsvforecasthub`, `covid19forecasthub`), so you can run it multiple times in a row without cleaning between runs.

#### R implementation (`external_to_projections.R`)

If you prefer R, run the companion script with the same flags:

```
Rscript scripts/external_to_projections.R \
    --output-path ./app/public/processed_data \
    --pathogen flu \
    --data-path /absolute/path/to/FluSight-forecast-hub/hub_data.csv \
    --target-data-path /absolute/path/to/FluSight-forecast-hub/target-data/time-series.csv \
    --locations-data-path /absolute/path/to/FluSight-forecast-hub/auxiliary-data/locations.csv \
    --overwrite
```

Swap the paths/pathogen exactly as in the Python example for RSV and COVID-19 Forecast Hubs (use `--pathogen rsvforecasthub` or `--pathogen covid19forecasthub`). Install dependencies once per machine:

```r
install.packages(c("jsonlite", "jsonvalidate", "arrow"))
```

The R script mirrors the Python pipeline: it validates inputs, converts them using the shared processing logic, and writes one JSON file per location (plus `metadata.json`) under `app/public/processed_data/<pathogen>/`. If the optional `jsonvalidate` package is installed, schema checks are applied; otherwise the script skips them with a warning.

## process_RespiLens_data

#### Overview

`process_RespiLens_data.py` is the command-line entry point for creating the full suite of RespiLens data required for a locally-hosted RespiLens site. Utilizing functionalities from `hub_dataset_processor.py`, `helper.py` and Hubverse's [hubData](https://hubverse-org.github.io/hub-data/) python package, this script will use paths to local clones of Hubverse hub repositories to process, convert, and generate JSON files that the RespiLens front end can use for visualization. 

#### Typical Workflow

1. User provides local hub path(s) (or the binary `--NHSN` flag, if processing NHSN data) and your desired output path via the command-line.
2. The correct processing path is determined based on which hub is currently being processed.
3. For each hub being processed:
    * Data is pre-processed (standardization) and target data/location metadata are retrieved from hub
    * Data is converted to RespiLens-style JSON
    * Data is saved to specified `--output-path` (pre-existing files in output directory will be overwritten)
If no hub path(s) are provided *and* `--NHSN` is not set, the script will exit. 

Example command:
```bash
# Process all data, run from RespiLens top-level
python scripts/process_RespiLens_data.py \
    --output-path ./app/public/processed_data
    --flusight-hub-path ./FluSight-forecast-hub \
    --covid-hub-path ./covid19-forecast-hub \
    --rsv-hub-path ./rsv-forecast-hub \
    --NHSN
```

#### Running `process_RespiLens_data.py`

You can run `process_RespiLens_data.py` with the following options:

| Option | Description | Value Type | Required? | Default Value |
| :--- | :--- | :--- | :--- | :--- |
| `--output-path` | Absolute path to directory where JSON files will be saved locally. | String | Yes | *N/A* |
| `--flusight-hub-path` | Absolute path to local clone of FluSight hub. | String | No | `None` |
| `--covid-hub-path` | Absolute path to local clone of COVID-19 hub. | String | No | `None` |
| `--rsv-hub-path` | Absolute path to local clone of RSV hub. | String | No | `None` |
| `--NHSN` | Flag for whether or not to process NHSN data. | boolean | No | `False` |

Alternatively, users can execute run the command `bash update_all_data_source.sh` from the top-level of the RespiLens directory to fetch/update all data required for local use of RespiLens.


## hub_dataset_processor

#### Overview

This script provides a standardized and reusable framework for transforming data from the Hubverse format into RespiLens projections-style JSON format. It standardizes input of model forecast data, ground truth data, and location metadata, then processes these into JSON files (one per location). It also produces a single `metadata.json` output file per set of a hub data conversion.

`hub_dataset_processor.py` is the foundational base for conversion, delegating specific implementation to one of `covid19_forecast_hub.py`, `rsv_forecast_hub.py`, or `flusight.py`. It is utilized internally via `process_RespiLens_data.py` and `external_to_projections.py`. 

## helper

#### Overview

`helper.py` contains a number of internal helper functions for use by the RespiLens python backend. Namely:

| Function | Utility | 
| :--- | :--- | 
| `clean_nan_values()` | Replaces `NaN` values of input dataframe to `None` (for JSON compatibility)
| `hubverse_data_preprocessor()` | Carries out a variety of pre-processing tasks for hubverse model data (data type standardization, value filtering, etc.) |
|  `get_location_info()` | Based on location metadata, retrieves a variety of location information using provided  FIPS code. |
| `save_json_file()` | Saves a JSON file to a specified output path (has modular overwriting settings) |
| `validate_respilens_json()` | Uses python `jsonschema` to validate JSON contents with the expected JSON schema of that type (either RespiLens 'projections' style or 'timeseries' style). |

It also contains three constants (`NHSN_COLUMN_MASKS`, `STATENAME_TO_ABBREVIATION_MAP`, and `STATEABBREVIATION_TO_FIPS_MAP`) for use by `nhsn_data_processor.py` and `nssp_data_processor.py`.


## nhsn_data_processor

#### Overview

While RSV, flu, and COVID-19 data are pulled to RespiLens from Hubverse, the NHSN view pulls from the CDC's [National Healthcare Safety Network](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data), and uses a different type of RespiLens JSON (hubverse data is converted to RespiLens projections JSON and NHSN data is converted to RespiLens timeseries JSON). `nhsn_data_processor.py` is triggered internally by `process_RespiLens_data.py`, at which point it sends an API request to a specific NHSN [resource](https://data.cdc.gov/resource/ua7e-t2fy.json), pulls the data in as JSON, then converts the payload to RespiLens timeseries-style JSON. As is the case with Hubverse data dumps, `nhsn_data_processor.py` creates one JSON file per location, and one `metadata.json` file per run. 


## processing

#### Overview

Within the `RespiLens/scripts/processing/` directory, you will find three scripts (`covid19_forecast_hub.py`, `rsv_forecast_hub.py`, and `flusight.py`) that provide hub-specific direction during the conversion process from Hubverse data -> RespiLens JSON. Each script contains a child implementation of the base class `hub_dataset_processor.HubDataProcessorBase` These files exist because of minute differences in hub setup that affect how we process data from each. 
