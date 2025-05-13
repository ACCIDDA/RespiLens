# RespiLens Data Format v2 (Draft)

*Last updated: May 6 2025*

---

## 1 · Rationale

RespiLens serves multiple respiratory‑disease datasets (FluSight, RSV Forecast Hub, NHSN, and others to come) inside one fast, URL‑shareable web app. To keep page‑loads snappy and code maintenance sane we standardise **two generic payload types**:

| Payload         | Purpose                                                | Core key    | Examples                                                        |
| --------------- | ------------------------------------------------------ | ----------- | --------------------------------------------------------------- |
| **Timeseries**  | Observed data and any longitudinal measurement series. | `series`    | NHSN admissions counts, FluSight ground‑truth, RSV ground‑truth |
| **Projections** | Forecasts / scenarios (point, quantile, sample, PMF).  | `forecasts` | Hub‑format weekly incident hosp forecasts                       |

Everything is built nightly by a GitHub Action, lives under `app/public/processed_data`, and is consumed by React + Plotly widgets.

---

## 2 · Top‑level layout

```
processed_data/
├── metadata.json          # global build + dataset catalogue
├── locations.json         # canonical list of locations (FIPS or ISO codes)
└── datasets/
    ├── flusight/
    │   ├── metadata.json  # FluSight‑specific options
    │   ├── timeseries/
    │   │   ├── US.json
    │   │   └── 01.json
    │   └── projections/
    │       ├── US.json
    │       └── 01.json
    ├── rsv_hub/
    │   └── …
    └── nhsn/
        └── …
```

(TODO: Ideally we would support parquet)

---

## 3 · Global `metadata.json`

```jsonc
{
  "build_timestamp": "2025‑05‑06T21:00:12Z",
  "datasets": {
    "flusight": {
      "fullName": "FluSight Hospitalisation Forecasts",
      "views": ["detailed", "timeseries"],
      "prefix": "flu",
      "hasModelSelector": true,
      "hasDateSelector": true
    },
    "rsv_hub": {«…»},
    "nhsn": {«…»}
  },
  "demo_mode": false
}
```

*Only fields the front‑end needs live here; anything dataset‑specific sits in that dataset’s own metadata file.*

---

## 4 · Canonical `locations.json`

Array of flat objects so every dataset can reference the same ids:

```jsonc
[
  {"location": "US", "abbreviation": "US", "name": "United States", "population": 333287557},
  {"location": "01", "abbreviation": "AL", "name": "Alabama", "population": 5039877},
  …
]
```

---

## 5 · Dataset‑level `metadata.json`

Example (flu):

```jsonc
{
  "shortName": "flusight",
  "fullName": "FluSight Hospitalisation Forecasts",
  "defaultView": "detailed",
  "targets": ["wk inc flu hosp", "wk flu hosp rate change", "peak week", "peak inc"],
  "quantile_levels": [0.025,0.25,0.5,0.75,0.975]
}
```

---

## 6 · Timeseries payload (`datasets/<ds>/timeseries/<loc>.json`)

```jsonc
{
  "metadata": {
    "dataset": "nhsn",
    "location": "NC",
    "series_type": "official"
  },
  "series": {
    "dates": ["2023‑10‑01", "2023‑10‑08", …],
    "columns": {
      "totalconfflunewadm": [120,145,…],
      "covid_new_adm": [45, 38, …]
    }
  }
}
```

*Notes*

* `series_type` enumerates variants (official, preliminary, etc.).
* Every numeric vector must align with `dates` index length.

---

## 7 · Projections payload (`datasets/<ds>/projections/<loc>.json`)

```jsonc
{
  "metadata": {
    "dataset": "flusight",
    "location": "NC"
  },
  "forecasts": {
    "2025‑02‑01": {
      "wk inc flu hosp": {
        "UNC_IDD‑influpaint": {
          "type": "quantile",
          "predictions": {
            "0": {"target_end_date": "2025‑02‑01", "quantiles": [0.025,0.25,0.5,0.75,0.975], "values": [100,120,130,140,160] },
            "1": {«…»}
          }
        },
        "FluSight‑ensemble": {«…»}
      }
    }
  }
}
```

*Differences from v1*

1. **Horizon keys remain strings** but target end date is explicit.
2. All prediction types (quantile | sample | pmf | point) live under the same schema; front‑end dispatches by `type`.
3. No mixing of targets and horizons—clears up nested merges.

---

## 8 · Conventions & validation

* ISO dates only.
* Horizons always strings “-1"‑"n".
* Quantile arrays strictly ascending matching `quantile_levels` in dataset metadata.
* Probabilities must sum to 1 within machine epsilon.
* Empty/missing elements are `[]`, never `null`.

---
