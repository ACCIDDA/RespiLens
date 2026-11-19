"""RespiLens processor for flu Metrocast Hubverse exports."""

import pandas as pd

from hub_dataset_processor import HubDataProcessorBase, HubDatasetConfig


class FluMetrocastDataProcessor(HubDataProcessorBase):
    def __init__(self, data: pd.DataFrame, locations_data: pd.DataFrame, target_data: pd.DataFrame,
                 peaks: dict | None = None):
        config = HubDatasetConfig(
            file_suffix="flu_metrocast",
            dataset_label="flu metrocast forecasts",
            ground_truth_min_date=pd.Timestamp("2024-08-01"),
        )
        # The frontend nests a sub-state location under its state by the ", XX" name suffix
        # (e.g. "NYC" -> "NYC, NY"); states have the code 'All'
        if "state_abb" in locations_data.columns:
            locations_data = locations_data.copy()
            needs_suffix = (locations_data["original_location_code"].astype(str) != "All") \
                & ~locations_data["location_name"].str.contains(",")
            locations_data.loc[needs_suffix, "location_name"] = (
                locations_data.loc[needs_suffix, "location_name"] + ", " + locations_data.loc[needs_suffix, "state_abb"]
            )
        super().__init__(
            data=data,
            locations_data=locations_data,
            target_data=target_data,
            config=config,
            is_metro_cast=True
        )
        # Peak forecasts of the long-term trajectories (see helper.samples_to_peaks), by location
        for location, location_peaks in (peaks or {}).items():
            payload = self.output_dict.get(f"{location}_{config.file_suffix}.json")
            if payload is not None and location_peaks:
                payload["peaks"] = location_peaks
