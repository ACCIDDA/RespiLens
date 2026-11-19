"""
Build the compact all-locations file behind the state summary's comparisons.

The per-state forecast files are 2-20 MB each, too heavy for a page to load
for every state. This reads the processed files once and keeps, for each
forecast date, disease and location, five numbers:

    [latest observed, ensemble median at horizons 0, 1, 2, 3]

The latest observation is the last one before the forecast date (what the
forecasters had), so horizons 1, 2 and 3 land 2, 3 and 4 weeks after it.
ED visits are percents; hospital admissions are per 100,000 people.
States come from the FluSight, COVID-19 and RSV hubs, sub-state locations
(health districts, metro areas) from Flu MetroCast.

Usage:
    python scripts/build_summary_snapshot.py \
        --data-path app/public/processed_data \
        [--flu-metrocast-hub-path flu-metrocast] \
        [--since 2025-08-01]
"""

import argparse
import csv
import glob
import json
import os

HORIZONS = ["0", "1", "2", "3"]

# disease -> (directory, file suffix, ensemble, ED target, hospital target)
HUBS = {
    "flu": ("flusight", "flu", "FluSight-ensemble",
            "wk inc flu prop ed visits", "wk inc flu hosp"),
    "covid": ("covid19forecasthub", "covid19", "CovidHub-ensemble",
              "wk inc covid prop ed visits", "wk inc covid hosp"),
    "rsv": ("rsvforecasthub", "rsv", "RSVHub-ensemble",
            "wk inc rsv prop ed visits", "wk inc rsv hosp"),
}

METROCAST_TARGET = "Flu ED visits pct"
# MetroCast's 4-week ensemble (named _mean in the 2025-26 season)
METROCAST_ENSEMBLES = ["epiENGAGE-ensemble_short", "epiENGAGE-ensemble_mean"]


def rounded(value):
    return None if value is None else float(f"{value:.4g}")


def latest_before(ground_truth, target, date):
    dates = ground_truth.get("dates", [])
    values = ground_truth.get(target, [])
    for d, v in zip(reversed(dates), reversed(values)):
        if d < date and v is not None:
            return v
    return None


def median(forecast, horizon):
    prediction = (forecast or {}).get("predictions", {}).get(horizon)
    if not prediction:
        return None
    quantiles = [float(q) for q in prediction.get("quantiles", [])]
    return prediction["values"][quantiles.index(0.5)] if 0.5 in quantiles else None


def row(ground_truth, target, date, forecast, scale):
    """[latest observed, medians at horizons 0-3], scaled; None if empty."""
    values = [latest_before(ground_truth, target, date)]
    values += [median(forecast, h) for h in HORIZONS]
    if all(v is None for v in values):
        return None
    return [rounded(v * scale) if v is not None else None for v in values]


def read_counties(hub_path):
    """Sub-state location -> the counties it covers, from the hub's list."""
    if not hub_path:
        return {}
    path = os.path.join(hub_path, "auxiliary-data", "locations.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {
            r["location"]: [c.strip() for c in r["hsa_counties"].split(",") if c.strip()]
            for r in csv.DictReader(f)
            if r.get("hsa_counties")
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-path", required=True,
                        help="The processed_data directory.")
    parser.add_argument("--flu-metrocast-hub-path",
                        help="Flu MetroCast hub clone, for each district's counties.")
    parser.add_argument("--since", default="2025-08-01",
                        help="First forecast date to keep (default: %(default)s).")
    args = parser.parse_args()

    snapshot = {"horizons": HORIZONS, "states": {}, "subLocations": {}, "dates": {}}

    def put(date, disease, kind, key, values):
        if values is not None and date >= args.since:
            (snapshot["dates"].setdefault(date, {})
             .setdefault(disease, {}).setdefault(kind, {}))[key] = values

    for disease, (directory, suffix, ensemble, ed_target, hosp_target) in HUBS.items():
        for path in sorted(glob.glob(os.path.join(args.data_path, directory, f"*_{suffix}.json"))):
            with open(path) as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            state = meta.get("abbreviation")
            population = meta.get("population")
            if not state or not population:
                continue
            snapshot["states"][state] = {"name": meta.get("location_name"), "population": population}
            ground_truth = data.get("ground_truth", {})
            for date, targets in data.get("forecasts", {}).items():
                put(date, disease, "ed", state,
                    row(ground_truth, ed_target, date, targets.get(ed_target, {}).get(ensemble), 100))
                put(date, disease, "hosp", state,
                    row(ground_truth, hosp_target, date, targets.get(hosp_target, {}).get(ensemble),
                        100000 / population))
        print(f"{disease}: done")

    counties = read_counties(args.flu_metrocast_hub_path)
    with open(os.path.join(args.data_path, "flumetrocast", "metadata.json")) as f:
        metrocast_locations = json.load(f).get("locations", [])
    for location in metrocast_locations:
        name = location.get("location_name", "")
        if ", " not in name:
            continue  # statewide entries are covered by the state rows
        key = location["abbreviation"]
        path = os.path.join(args.data_path, "flumetrocast", f"{key}_flu_metrocast.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        snapshot["subLocations"][key] = {
            "name": name.rsplit(", ", 1)[0],
            "state": name.rsplit(", ", 1)[1],
            "population": location.get("population"),
            "counties": counties.get(key, []),
        }
        for date, targets in data.get("forecasts", {}).items():
            models = targets.get(METROCAST_TARGET, {})
            forecast = next((models[m] for m in METROCAST_ENSEMBLES if m in models), None)
            put(date, "flu", "edSub", key,
                row(data.get("ground_truth", {}), METROCAST_TARGET, date, forecast, 1))
    print("metrocast: done")

    out_dir = os.path.join(args.data_path, "summary")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "snapshot.json")
    with open(out_path, "w") as f:
        json.dump(snapshot, f, separators=(",", ":"))
    print(f"Wrote {out_path} ({os.path.getsize(out_path) // 1024} KB, "
          f"{len(snapshot['dates'])} dates)")


if __name__ == "__main__":
    main()
