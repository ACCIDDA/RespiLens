# RespiLens ![respilens-logo](https://github.com/user-attachments/assets/f4b54c2a-9d27-4453-9a85-72b1b2f965a2)

https://accidda.github.io/RespiLens/ (formerly RespiView)

Hey, there are other flusight vizualization but realized most of them are geared towards CDC/academics instead of state health department and public. Moreover, these hub data remains not accesssible enough. The goal is to make one more with these users in mind, and to have the following important (to me only) features:
- **ability to link a certain view to a URL to share forecast**
- static and updated everyweek
- multi view (categorical forecasts, scores, timeseries)
- multi-date on one view to compare
- any number of model


Most other viz I am aware off are made by the amazing Reichlab:
- http://flusightnetwork.io: very great, but only past season and only one model somehow
- https://zoltardata.com/project/360/viz the zotltra view is only past season but is very great, I like the suffle color button. Only last year
- https://reichlab.io/flusight-dashboard/ very detailed dashboard, incredible for modelers. Currently 404 and also it is relatively static for other users.

## Features
Features
### TODO
- [x] Add the RSV forecasts
- [x] Add the multiview view
- [ ] Add a multipathogen view
- [ ] Show on state card which one have rsv data
- [x] gray rsv view on state with no rsv
- [ ] direct link to plot png/pdf ?
- [x] Simple view for mobile
- [x] add some text to welcome and disclaim
- [ ] model description when we hover on name
- [x] validation plot for rsv
- [ ] Add WIS score and
- [ ] allow to select top scoring model past 8 week or full past for this state
- [ ] logo as favicon
- [ ] change current json payload format to binary format
- [ ] reorder URL: state then view then date then models
- [x] hold model and date on state change
- [x] deploy to ghpage using action instead of branch
- [x] view without histogram for mobile viewing
- [ ] remove histogram on multidate view cause confusing
- [ ] add peak timing and size targets

### DONE

- [x] basic forecast visualization component
- [x] set up github actions workflow for data processing
- [x] add state/location selector
- [x] implement model selection
- [x] add date navigation controls
- [x] create plotly-based interactive charts
- [x] add quantile confidence intervals visualization
- [x] implement rate change histogram
- [x] add model color coding system
- [x] enable plot export functionality
- [x] add responsive layout design
- [x] implement url parameter handling
- [x] add select all/none model buttons
- [x] create multi-date selection feature
- [x] add reset view functionality
- [x] implement plot navigation timeline
- [x] add development proxy configuration
- [x] set up github pages deployment
- [x] implement data processing pipeline
- [x] add progress tracking for data processing
- [x] create validation plots
- [x] implement ground truth data filtering
- [x] add automatic metadata generation
- [x] choose name  **RespiLense better ??**
- [x] make logo
- [x] implement error handling for data loading
- [x] create mobile-friendly interface
- [x] add documentation for data formats
- [x] create consistent file naming system



## Desired Behaviors

There are 3 dataset and many view, some per dataset some multidateset
* flusight with shortkey: flu
* rsvforecasthub with shortkey rsv
* nhsn with shortkey nhsn

View switching within a dataset (e.g Flu detailed to Flu Timeseries) should preserve all url Parameters
* View switching should between dataset should reset all url parameters to be empty and the plot to be the default

* URL Parameters Format:
  * location: State abbreviation (e.g. "MA")
  * view: "fludetailed", "flutimeseries", or "rsvdetailed", or "nhsnall"

For flu views:
  * flu_dates: Comma-separated dates for flu views
  * flu_models: Comma-separated model names for flu views
For rsv views:
  * rsv_dates: Comma-separated dates for RSV view
  * rsv_models: Comma-separated model names for RSV view
For nshn views:
  * prelim_col: column to display on the preliminary dataset
  * final_col: column on the official dataset

Parameter Logic:
  * URL params are dataset specific (flu_ or rsv_ prefix)
  * Switching between flu views preserves flu_* params
  * Switching between RSV/flu clears old view's params
Reset clears current view's params and sets defaults. 

Default for flu and rsv views: 
  *  Most recent date
  *  Default ensemble model plotted (FluSight/hub based on view)
  *  Default axis (8 week before, 5 week after the last date.)
  *  model selector show possible model

Default for NHSN views
* Two model selector show columns (for prelim and final data)
* no date selector


Location and view params always preserved



# FluSight Visualization Data Format

This document describes the data format used for the FluSight visualization payloads. The preprocessed data is organized into multiple JSON files: a metadata file and individual location files.

## Directory Structure

```
processed_data/
├── metadata.json         # Global metadata and available options
├── US.json              # National level forecasts and data
├── 01.json             # Alabama forecasts and data
├── 02.json             # Alaska forecasts and data
└── ...                 # Other state/territory files
```

## Metadata File Format

The `metadata.json` file contains global information about the available data:

```json
{
  "last_updated": "2024-01-18 15:30:00",
  "models": ["UNC_IDD-influpaint", "FluSight-ensemble", ...],
  "locations": ["US", "01", "02", ...],
  "demo_mode": false
}
```

| Field | Description |
|-------|-------------|
| `last_updated` | Timestamp of when the data was last processed |
| `models` | Array of available model identifiers |
| `locations` | Array of available location codes |
| `demo_mode` | Boolean indicating if this is demo data (limited models) |

## Location File Format

Each location file (e.g., `US.json`, `01.json`) contains three main sections:

```json
{
  "metadata": {
    "location": "01",
    "location_name": "Alabama",
    "population": 5024279
  },
  "ground_truth": {
    "dates": ["2023-10-01", "2023-10-08", ...],
    "values": [120, 145, ...],
    "rates": [2.4, 2.9, ...]
  },
  "forecasts": {
    "2024-01-13": {
      "wk inc flu hosp": {
        "UNC_IDD-influpaint": {
          "type": "quantile",
          "predictions": {
            "0": {
              "date": "2024-01-13",
              "quantiles": [0.025, 0.25, 0.5, 0.75, 0.975],
              "values": [100, 120, 130, 140, 160]
            },
            "1": {
              "date": "2024-01-20",
              "quantiles": [0.025, 0.25, 0.5, 0.75, 0.975],
              "values": [110, 130, 140, 150, 170]
            }
          }
        }
      },
      "wk flu hosp rate change": {
        "UNC_IDD-influpaint": {
          "type": "pmf",
          "predictions": {
            "0": {
              "date": "2024-01-13",
              "categories": ["large_decrease", "decrease", "stable", "increase", "large_increase"],
              "probabilities": [0.1, 0.2, 0.4, 0.2, 0.1]
            }
          }
        }
      }
    }
  }
}
```

### Metadata Section

Location-specific metadata:

| Field | Description |
|-------|-------------|
| `location` | Location code (e.g., "01" for Alabama) |
| `location_name` | Human-readable location name |
| `population` | Population size used for rate calculations |

### Ground Truth Section

Historical observed data:

| Field | Description |
|-------|-------------|
| `dates` | Array of dates in YYYY-MM-DD format |
| `values` | Array of observed hospitalization counts |
| `rates` | Array of hospitalization rates per 100k population |

### Forecasts Section

The forecasts section is organized hierarchically:

1. Reference Date (YYYY-MM-DD)
   - The Saturday of the submission week
2. Target Type
   - `wk inc flu hosp`: Weekly incident hospitalizations
   - `wk flu hosp rate change`: Weekly rate change categories
   - `peak week inc flu hosp`: Peak week prediction
   - `peak inc flu hosp`: Peak incidence prediction
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

#### Sample Predictions Format

Used for trajectory samples:

```json
{
  "type": "sample",
  "predictions": {
    "0": {
      "date": "2024-01-13",
      "samples": [120, 125, 130, ...]
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `type` | Always "sample" |
| `predictions` | Object with horizons as keys (0-3) |
| `date` | Target end date for the prediction |
| `samples` | Array of sample values |

## Usage Notes

1. All dates use ISO format (YYYY-MM-DD)
2. Horizons are stored as strings ("0", "1", "2", "3")
3. Values for incident hospitalizations are always integers
4. PMF probabilities always sum to 1.0
5. Sample trajectories contain exactly 100 samples per horizon
6. Missing data is represented by empty arrays or null values
