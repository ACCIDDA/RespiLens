"""RespiLens processor for flu Metrocast Hubverse exports."""

import numpy as np
import pandas as pd

from hub_dataset_processor import HubDataProcessorBase, HubDatasetConfig

SAMPLE_TRAJECTORY_COUNT = 30
SAMPLE_TRAJECTORY_SEED = 2026


class FluMetrocastDataProcessor(HubDataProcessorBase):
    def __init__(
        self,
        data: pd.DataFrame,
        locations_data: pd.DataFrame,
        target_data: pd.DataFrame,
        sample_rows: pd.DataFrame | None = None,
    ):
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
        if sample_rows is not None and not sample_rows.empty:
            self._add_sample_trajectories(sample_rows)

    def _add_sample_trajectories(self, samples: pd.DataFrame) -> None:
        """Attach up to 30 complete submitted paths per date, target, and model."""
        samples = samples[samples['location'].isin(self.locations_in_this_dump)]
        group_keys = ['location', 'reference_date', 'target', 'model_id']
        for (location, reference_date, target, model), group in samples.groupby(group_keys, observed=True):
            payload = self.output_dict[f'{location}_{self.config.file_suffix}.json']
            date = pd.Timestamp(reference_date).date().isoformat()
            if model not in payload['forecasts'].get(date, {}).get(target, {}):
                continue

            trajectories = group.pivot_table(
                index='output_type_id', columns='target_end_date', values='value',
                aggfunc='first', observed=True, dropna=False,
            ).sort_index().sort_index(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if trajectories.empty:
                continue

            rng = np.random.default_rng(SAMPLE_TRAJECTORY_SEED)
            selected = rng.choice(
                len(trajectories), size=min(SAMPLE_TRAJECTORY_COUNT, len(trajectories)), replace=False,
            )
            by_date = payload.setdefault('sample_trajectories', {}).setdefault(date, {})
            by_date.setdefault(target, {})[model] = {
                'dates': [pd.Timestamp(end_date).date().isoformat() for end_date in trajectories.columns],
                'samples': [[round(float(value), 2) for value in row]
                            for row in trajectories.iloc[sorted(selected)].to_numpy()],
            }
