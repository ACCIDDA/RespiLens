import json
import sys
import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from external_data import load_inputs
from processors import FlusightDataProcessor, FluMetrocastDataProcessor, RSVDataProcessor, COVIDDataProcessor
from helper import hubverse_df_preprocessor, samples_to_quantiles


def _load_expected(filename: str) -> dict:
    path = Path(__file__).resolve().parent / "samples" / "expected" / filename
    with path.open() as handle:
        return json.load(handle)


def _sanitize(payload: dict) -> dict:
    payload = json.loads(json.dumps(payload))  # deep copy
    metadata = payload.get("metadata", {})
    metadata.pop("last_updated", None)
    return _canonicalize_numbers(payload)


def _canonicalize_numbers(obj):
    if isinstance(obj, dict):
        return {key: _canonicalize_numbers(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_canonicalize_numbers(value) for value in obj]
    if isinstance(obj, float):
        return int(obj) if obj.is_integer() else round(obj, 10)
    return obj


def test_flusight_processor_matches_expected():
    base = Path(__file__).resolve().parent / "samples" / "flusight"
    inputs = load_inputs(
        pathogen="flu",
        data_path=base / "forecast_data.csv",
        target_data_path=base / "target_data.csv",
        locations_data_path=base / "locations.csv",
    )

    processor = FlusightDataProcessor(
        data=inputs.data,
        locations_data=inputs.locations_data,
        target_data=inputs.target_data,
    )

    actual = _sanitize(processor.output_dict["CA_flu.json"])
    expected = _sanitize(_load_expected("CA_flu.json"))
    assert actual == expected

    actual_meta = _sanitize(processor.output_dict["metadata.json"])
    expected_meta = _sanitize(_load_expected("metadata.json"))
    assert actual_meta["models"] == expected_meta["models"]
    assert actual_meta["locations"] == expected_meta["locations"]


@pytest.mark.parametrize("processor_class,suffix", [
    (FlusightDataProcessor, "flu"),
    (RSVDataProcessor, "rsv"),
    (COVIDDataProcessor, "covid19"),
    (FluMetrocastDataProcessor, "flu_metrocast"),
])
@pytest.mark.parametrize("output_type", ["quantile", "pmf", "sample"])
def test_negative_horizons_survive_json_export(processor_class, suffix, output_type):
    if output_type == "sample" and processor_class is not FluMetrocastDataProcessor:
        pytest.skip("Only Metrocast converts sample trajectories to quantiles")

    reference_date = datetime.date(2026, 10, 7)
    horizons = [-2, -1, 0, 1]
    target = "wk inc flu hosp"
    forecasts = pd.DataFrame([
        {
            "location": "06", "reference_date": reference_date,
            "target": target, "model_id": "test-model",
            "horizon": None if output_type == "sample" else horizon,
            "output_type": output_type,
            "output_type_id": "stable" if output_type == "pmf" else 0.5,
            "value": 0.8 if output_type == "pmf" else 42,
            "target_end_date": reference_date + datetime.timedelta(days=3 + 7 * horizon),
        }
        for horizon in horizons
    ])
    if output_type == "sample":
        forecasts = samples_to_quantiles(forecasts)
    forecasts = hubverse_df_preprocessor(forecasts, filter_nowcasts=False)
    locations = pd.DataFrame([{
        "location": "06", "abbreviation": "CA", "location_name": "California",
        "original_location_code": "06", "population": 1000,
    }])
    truth = pd.DataFrame([
        {"location": "06", "target": target, "as_of": "2026-10-14",
         "target_end_date": reference_date + datetime.timedelta(days=3 + 7 * horizon),
         "observation": 40}
        for horizon in horizons
    ])

    processor = processor_class(forecasts, locations, truth)
    abbreviation = "06" if processor_class is FluMetrocastDataProcessor else "CA"
    # Exercise the actual JSON representation consumed by the charts.
    payload = json.loads(json.dumps(processor.output_dict[f"{abbreviation}_{suffix}.json"]))
    assert set(payload["metadata"]["hubverse_keys"]["horizons"]) == {str(h) for h in horizons}
    predictions = payload["forecasts"][str(reference_date)][target]["test-model"]["predictions"]
    assert set(predictions) == {str(h) for h in horizons}
    for horizon in horizons:
        prediction = predictions[str(horizon)]
        assert prediction["date"] == str(reference_date + datetime.timedelta(days=3 + 7 * horizon))
        assert prediction["date"] in payload["ground_truth"]["dates"]
        if output_type == "pmf":
            assert prediction["categories"] == ["stable"]
            assert prediction["probabilities"] == [0.8]
        else:
            assert 0.5 in prediction["quantiles"]
            assert prediction["values"] == [42] * len(prediction["quantiles"])


def test_nowcast_filter_remains_optional():
    forecasts = pd.DataFrame({
        "horizon": [-2, -1, 0, 1], "target": ["wk inc flu hosp"] * 4,
        "output_type": ["quantile"] * 4, "output_type_id": [0.5] * 4,
    })
    assert hubverse_df_preprocessor(forecasts, filter_nowcasts=True)["horizon"].tolist() == [0, 1]
    assert hubverse_df_preprocessor(forecasts, filter_nowcasts=False)["horizon"].tolist() == [-2, -1, 0, 1]


def test_metrocast_exports_complete_deterministic_sample_trajectories():
    reference_date = datetime.date(2026, 10, 31)
    target = "Flu ED visits pct"
    dates = [reference_date, reference_date + datetime.timedelta(weeks=1)]
    samples = pd.DataFrame([
        {
            "location": "denver", "reference_date": reference_date,
            "target": target, "model_id": model, "target_end_date": dates[week],
            "horizon": None, "output_type": "sample", "output_type_id": f"S{sample}",
            "value": offset + sample + week * 0.25,
        }
        for model, offset, count in [("long", 0, 40), ("short", 100, 2)]
        for sample in range(count)
        for week in range(2)
        if (model, sample, week) != ("long", 0, 1)
    ])
    forecasts = hubverse_df_preprocessor(samples_to_quantiles(samples), filter_nowcasts=False)
    locations = pd.DataFrame([{
        "location": "denver", "original_location_code": "688",
        "location_name": "Denver, CO", "population": 2973329,
    }])
    truth = pd.DataFrame([{
        "location": "denver", "target": target, "as_of": dates[1],
        "target_end_date": dates[0], "observation": 40,
    }])
    payload = FluMetrocastDataProcessor(forecasts, locations, truth, samples).output_dict["denver_flu_metrocast.json"]
    shuffled = samples.sample(frac=1, random_state=7)
    repeated = FluMetrocastDataProcessor(forecasts, locations, truth, shuffled).output_dict["denver_flu_metrocast.json"]

    overlay = payload["sample_trajectories"]
    assert overlay == repeated["sample_trajectories"]
    models = overlay[reference_date.isoformat()][target]
    assert models["long"]["dates"] == [date.isoformat() for date in dates]
    paths = models["long"]["samples"]
    assert len({tuple(path) for path in paths}) == len(paths) == 30
    assert all(1 <= start < 40 and end == start + 0.25 for start, end in paths)
    assert models["short"]["samples"] == [[100, 100.25], [101, 101.25]]
    first_week = payload["forecasts"][reference_date.isoformat()][target]["long"]["predictions"]["0"]
    assert first_week["values"][first_week["quantiles"].index(0.5)] == 19.5


@pytest.mark.parametrize("include_current_forecast", [True, False])
def test_metrocast_uses_current_hub_locations(include_current_forecast):
    rows = [
        ("ogden", datetime.date(2026, 10, 7), "retired-model"),
        ("mesa", datetime.date(2025, 11, 12), "old-model"),
    ]
    if include_current_forecast:
        rows.append(("denver", datetime.date(2026, 10, 7), "current-model"))
    forecasts = pd.DataFrame([
        {
            "location": location, "reference_date": reference_date,
            "target": "wk inc flu hosp", "model_id": model,
            "horizon": 0, "output_type": "quantile", "output_type_id": 0.5,
            "value": 42, "target_end_date": datetime.date(2026, 10, 10),
        }
        for location, reference_date, model in rows
    ])
    locations = pd.DataFrame([
        {"location": "denver", "original_location_code": "688",
         "location_name": "Denver, CO", "population": 2973329},
        {"location": "mesa", "original_location_code": "711",
         "location_name": "Mesa, CO", "population": 299744},
    ])
    truth = pd.DataFrame([
        {"location": location, "target": "wk inc flu hosp",
         "as_of": "2026-10-14", "target_end_date": "2026-10-10",
         "observation": 40}
        for location in ["denver", "ogden"]
    ])

    processor = FluMetrocastDataProcessor(forecasts, locations, truth)
    metadata = processor.output_dict["metadata.json"]

    if include_current_forecast:
        assert set(processor.output_dict) == {"metadata.json", "denver_flu_metrocast.json"}
        assert metadata["models"] == ["current-model"]
        assert [loc["abbreviation"] for loc in metadata["locations"]] == ["denver"]
        payload = processor.output_dict["denver_flu_metrocast.json"]
        assert payload["metadata"]["location"] == "688"
        assert payload["metadata"]["population"] == 2973329
        assert payload["ground_truth"] == {
            "dates": ["2026-10-10"], "wk inc flu hosp": [40]
        }
        prediction = payload["forecasts"]["2026-10-07"]["wk inc flu hosp"]["current-model"]
        assert prediction["predictions"]["0"] == {
            "date": "2026-10-10", "quantiles": [0.5], "values": [42]
        }
    else:
        assert set(processor.output_dict) == {"metadata.json"}
        assert metadata["models"] == []
        assert metadata["locations"] == []

    # Filtering does not mutate the caller's forecasts or location table.
    assert "ogden" in set(forecasts["location"])
    assert set(locations["location"]) == {"denver", "mesa"}
