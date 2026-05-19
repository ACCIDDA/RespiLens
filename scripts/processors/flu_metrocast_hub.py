"""RespiLens processor for flu Metrocast Hubverse exports."""

import pandas as pd

from hub_dataset_processor import HubDataProcessorBase, HubDatasetConfig


class FluMetrocastDataProcessor(HubDataProcessorBase):
    def __init__(self, data: pd.DataFrame, locations_data: pd.DataFrame, target_data: pd.DataFrame):
        config = HubDatasetConfig(
            file_suffix="flu_metrocast",
            dataset_label="flu metrocast forecasts",
            ground_truth_min_date=pd.Timestamp("2024-08-01"),
        )
        super().__init__(
            data=data,
            locations_data=locations_data,
            target_data=target_data,
            config=config,
            is_metro_cast=True
        )