"""RespiLens processor for the RSV Forecast Hub."""

import pandas as pd

from hub_dataset_processor import HubDataProcessorBase, HubDatasetConfig


class RSVDataProcessor(HubDataProcessorBase):
    def __init__(self, data: pd.DataFrame, locations_data: pd.DataFrame, target_data: pd.DataFrame):
        config = HubDatasetConfig(
            file_suffix="rsv",
            dataset_label="rsv forecast hub",
            ground_truth_min_date=pd.Timestamp("2023-10-01"),
        )
        super().__init__(
            data=data,
            locations_data=locations_data,
            target_data=target_data,
            config=config,
        )
