"""RespiLens processor for FluSight Hubverse exports."""

import pandas as pd

from hub_dataset_processor import HubDataProcessorBase, HubDatasetConfig


class FlusightDataProcessor(HubDataProcessorBase):
    def __init__(self, data: pd.DataFrame, locations_data: pd.DataFrame, target_data: pd.DataFrame):
        config = HubDatasetConfig(
            file_suffix="flu",
            dataset_label="flusight forecasts",
            ground_truth_min_date=pd.Timestamp("2022-10-01"),
        )
        super().__init__(
            data=data,
            locations_data=locations_data,
            target_data=target_data,
            config=config,
        )
