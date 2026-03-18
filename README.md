# [RespiLens.com](http://www.respilens.com) [![Lint](https://github.com/ACCIDDA/RespiLens-staging/actions/workflows/lint.yml/badge.svg)](https://github.com/ACCIDDA/RespiLens-staging/actions/workflows/lint.yml)  [![Build and Deploy Site](https://github.com/ACCIDDA/RespiLens-staging/actions/workflows/build-deploy.yml/badge.svg)](https://github.com/ACCIDDA/RespiLens-staging/actions/workflows/build-deploy.yml)  [![Python Tests](https://github.com/ACCIDDA/RespiLens-staging/actions/workflows/python-tests.yml/badge.svg)](https://github.com/ACCIDDA/RespiLens-staging/actions/workflows/python-tests.yml)
Authors: **Emily Przykucki**, Joseph Lemaitre, and others within ACCIDDA, the Atlantic Coast Center for Infectious Disease Dynamics and Analytics.

* **stable version** https://www.RespiLens.com
* **new features are developed on** https://staging.RespiLens.com, [GitHub](https://github.com/ACCIDDA/RespiLens-staging)

RespiLens is a responsive web app to visualize respiratory disease forecasts in the US, focused on accessibility for state health departments and the general public. The RSV, COVID-19 and flu views pull from CDC forecast hubs (collectively known as the Hubverse: [rsv-forecast-hub](https://github.com/CDCgov/rsv-forecast-hub), [covid19-forecast-hub](https://github.com/CDCgov/covid19-forecast-hub), and [FluSight-forecast-hub](https://github.com/cdcepi/FluSight-forecast-hub)), and consolidate pathogen data by location and date into one user-friendly plot. While other visualization tools for FluSight/RSV/COVID-19 Hubverse data may exist, many of them are geared towards academics instead of state health departments and the public, making them less accessible to people who need them. Our goal is to make a dashboard with these users in mind. Presently, RespiLens offers these features:
- **Ability to link a certain view to a URL to share a forecast.** Select a pathogen, a location, and date(s), and then click the "Share View" button to copy the URL with your settings.
- Site rebuilt daily, with new data Hub produced weekly during Hub forecasting seasons (year-round for COVID-19)
- Ability to choose any number of dates to visualize on your plot
- Ability to choose any number of contributing models to visualize on your plot
- A view with National Healthcare Safety Network data, plotting almost 300 data categories (e.g., Number of Adult COVID-19 Admissions, or Percent Inpatient Beds Occupied by RSV Patients) 
- **Forecastle**, a daily disease forecasting game, where you can attempt to predict hospitalizations counts for a specific scenario and then get scored against participating models.
- **MyRespiLens**, where you can visualize your own data for a specific location without the file leaving your machine.


## Data Processing

GitHub Actions refreshes (for the website) the processed JSON nightly, but you can regenerate all data locally with a single executable:

- `./update_all_data_source.sh`  
  Run from the top-level of your local RespiLens directory. This clones and/or updates NHSN data, and repositories for FluSight, RSV, and COVID-19 hub data, then runs `scripts/process_RespiLens_data.py` to convert data to RespiLens-sanctioned formats and place the output in `app/public/processed_data/` (where the js codebase pulls site data from).

- `python scripts/external_to_projections.py`  
  Converts a single pathogen’s Hubverse .csv data into RespiLens projections-style JSON. Supply the path to the .csv, target dataset, and location metadata for the pathogen you want to process, and run (e.g.) the following from the top-level of your local RespiLens directory

  ```bash
  python scripts/external_to_projections.py \
      --output-path ./path/to/your/output_directory \
      --pathogen flu \
      --data-path /abs_path_to/FluSight-forecast-hub/model-output/.../loc.csv \
      --target-data-path /absolute/path/to/FluSight-forecast-hub/target-data/time-series.csv \
      --locations-data-path /absolute/path/to/FluSight-forecast-hub/auxiliary-data/locations.csv \
      --overwrite
  ```

- `Rscript scripts/external_to_projections.R`  
  The R companion script accepts the same arguments and produces byte-for-byte identical JSON (after installing `jsonlite` and `arrow`). If `jsonvalidate` is available it will also run schema checks, otherwise it skips them with a warning. Use it when working entirely in the R ecosystem or Hubverse R tooling. The command looks almost identical to that of the Python pipeline:

  ```bash
  Rscript scripts/external_to_projections.R \
      --output-path ./path/to/your/output_directory \
      --pathogen flu \
      --data-path /abs_path_to/FluSight-forecast-hub/model-output/.../loc.csv \
      --target-data-path /absolute/path/to/FluSight-forecast-hub/target-data/time-series.csv \
      --locations-data-path /absolute/path/to/FluSight-forecast-hub/auxiliary-data/locations.csv \
      --overwrite
  ```

JSON files produced using `external_to_projections.py` or `external_to_projections.R` can be drag-n-dropped to MyRespiLens. If you want to compare fresh output against a snapshot, stash the reference JSON (e.g., under `tmp/baseline_json_samples/`) and run:

```bash
diff -ru tmp/baseline_json_samples app/public/processed_data
```

For a quick regression check that the Python and R converters remain in sync, run the bundled parity script:

```bash
./tests/run_projection_parity_test.sh
```

Details live in `tests/README.md`.

Unit tests that exercise the shared processors against a fixed FluSight fixture are available via:

```bash
python -m pytest tests/test_processors.py
```


### Metadata

Location-specific metadata (appears in Hubverse hubs as `locations.csv`):

| Field | Description |
|-------|-------------|
| `abbreviation` | Abbreviation for the location (e.g., "NC" for North Carolina) |
| `location` | Location FIPS code (e.g., "37" for North Carolina) |
| `location_name` | Human-readable location name (e.g., "North Carolina" for North Carolina) |
| `population` | Population size |

Each RespiLens JSON data file has a `metadata` key, containing details about the contents of the file. RespiLens projections data (RSV, COVID-19, and flu) metadata keys will resemble this:

```json
{
  "metadata": {
    "location": "37",
    "abbreviation": "NC",
    "location_name": "North Carolina",
    "population": 10488084,
    "dataset": "covid19 forecast hub",
    "series_type": "projection",
    "hubverse_keys": {
      "models": ["model_name1", "model_name2"],
      "targets": ["wk inc covid hosp", "wk inc covid prop ed visits"],
      "horizons": ["0", "1", "2"],
      "output_types": ["quantile"]
    }
  }
}
```
FluSight data contains an additional "pmf" data output type, which RespiLens uses to create a histogram of hospitalization categories (large increase, increase, stable, decrease, large decrease) for a given horizon. This can be viewed by selecting the FluSight Detailed View ([see below for more information](#pmf-predictions-format))

And RespiLens timeseries data (NHSN) metadata keys will resemble this:

```json
{
  "metadata": {
    "location": "37",
    "abbreviation": "NC",
    "location_name": "North Carolina",
    "population": 10488084,
    "dataset": "NHSN",
    "series_type": "projection"
  }
}
```

Additionally, each data pull will produce one `metadata.json` file stored amongst location JSON output, which consolidates information about locations included in that pull, as well as what date/time the pull occurred.

### Ground Truth 

The `ground_truth` key of RespiLens projections JSON stores the actual, observed data for a specific pathogen.

| Field | Description |
|-------|-------------|
| `dates` | Array of dates in YYYY-MM-DD format |
| `<target_name>` | Array of observed values for that target |

Targets from the Hubverse are often "wk inc hosp", meaning "weekly incident of hospitalization" for a certain pathogen. Some pathogens (RSV and COVID-19) contain a "proportion of ED visits" due to a certain pathogen target as well. 

All NHSN data is ground truth (actually observed).

### Forecasts Section

The forecasts section is organized hierarchically:

1. Reference Date (YYYY-MM-DD)
   - The Saturday of the submission week
2. Target 
   - The target value forecasted (e.g., "weekly incidence of hospitalization")
3. Model Name
   - Contains predictions for each model
4. Prediction Data
   - Type-specific format for each kind of prediction

#### Quantile Predictions Format

Used for incident hospitalizations:

```json
{
  "type": "quantile",
  "predictions": {
    "0": {
      "date": "2024-01-13",
      "quantiles": [0.025, 0.25, 0.5, 0.75, 0.975],
      "values": [100, 120, 130, 140, 160]
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `type` | Always "quantile" |
| `predictions` | Object with horizons as keys (0-3) |
| `date` | Target end date for the prediction |
| `quantiles` | Array of quantile levels |
| `values` | Array of predicted values for each quantile |

#### PMF Predictions Format

Used for rate change categories:

```json
{
  "type": "pmf",
  "predictions": {
    "0": {
      "date": "2024-01-13",
      "categories": ["large_decrease", "decrease", "stable", "increase", "large_increase"],
      "probabilities": [0.1, 0.2, 0.4, 0.2, 0.1]
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `type` | Always "pmf" |
| `predictions` | Object with horizons as keys (0-3) |
| `date` | Target end date for the prediction |
| `categories` | Array of category names |
| `probabilities` | Array of probability values summing to 1 |

## Usage Notes

1. All dates use ISO format (YYYY-MM-DD)
2. Horizons and FIPS codes are stored as strings ("0", "1", "2"; "01", "37", "50")
3. Values for incident hospitalizations are always integers
4. PMF probabilities always sum to 1.0
6. Missing data is represented by JSON-compatible null values
