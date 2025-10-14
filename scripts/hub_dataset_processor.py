"""
Shared utilities for processing Hubverse forecast datasets into RespiLens JSON.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging

import pandas as pd

from helper import get_location_info


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HubDatasetConfig:
    """Configuration describing dataset-specific processing details."""

    file_suffix: str
    dataset_label: str
    ground_truth_date_column: str
    ground_truth_min_date: Optional[pd.Timestamp] = None
    series_type: str = "projection"
    observation_column: str = "observation"
    drop_output_types: Tuple[str, ...] = ("sample",)


class HubDataProcessorBase:
    """
    Base processor that handles the shared JSON export workflow for Hubverse datasets.

    Subclasses supply dataset-specific configuration via HubDatasetConfig.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        locations_data: pd.DataFrame,
        target_data: pd.DataFrame,
        config: HubDatasetConfig,
    ) -> None:
        self.output_dict: Dict[str, Dict[str, Any]] = {}
        self.df_data = data
        self.locations_data = locations_data
        self.target_data = target_data
        self.config = config

        self.logger = logging.getLogger(self.__class__.__name__)
        self.location_dataframes: Dict[str, pd.DataFrame] = {}
        self.ground_truth_dataframes: Dict[str, pd.DataFrame] = {}

        self.logger.info("Building individual %s JSON files...", self.config.dataset_label)
        self._build_outputs()

        metadata_file_contents = self._build_metadata_file(self._build_all_models_list())
        self.output_dict["metadata.json"] = metadata_file_contents
        self.logger.info("Success ✅")

        # Expose a consolidated dictionary of intermediate DataFrames for future exports.
        self.intermediate_dataframes: Dict[str, Any] = {
            "hubverse_raw": self.df_data,
            "locations": self.location_dataframes,
            "ground_truth": self.ground_truth_dataframes,
        }

    def _build_outputs(self) -> None:
        """Create per-location JSON payloads."""
        locations_gbo = self.df_data.groupby("location")
        for loc, loc_df in locations_gbo:
            loc_str = str(loc)
            loc_df = loc_df.copy()
            self.location_dataframes[loc_str] = loc_df

            location_abbreviation = get_location_info(
                location_data=self.locations_data, location=loc_str, value_needed="abbreviation"
            )
            file_name = f"{location_abbreviation}_{self.config.file_suffix}.json"

            ground_truth_df = self._prepare_ground_truth_df(location=loc_str)
            self.ground_truth_dataframes[loc_str] = ground_truth_df.copy()

            metadata = self._build_metadata_key(df=loc_df)
            ground_truth = self._format_ground_truth_output(ground_truth_df=ground_truth_df)
            forecasts = self._build_forecasts_key(df=loc_df)

            self.output_dict[file_name] = {
                "metadata": metadata,
                "ground_truth": ground_truth,
                "forecasts": forecasts,
            }

    def _build_metadata_key(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build metadata section of an individual JSON file."""
        location = str(df["location"].iloc[0])
        metadata = {
            "location": location,
            "abbreviation": get_location_info(
                self.locations_data, location=location, value_needed="abbreviation"
            ),
            "location_name": get_location_info(
                self.locations_data, location=location, value_needed="location_name"
            ),
            "population": get_location_info(
                self.locations_data, location=location, value_needed="population"
            ),
            "dataset": self.config.dataset_label,
            "series_type": self.config.series_type,
            "hubverse_keys": {
                "models": self._build_available_models_list(df=df),
                "targets": list(dict.fromkeys(df["target"])),
                "horizons": [str(h) for h in pd.unique(df["horizon"])],
                "output_types": [
                    item for item in pd.unique(df["output_type"]) if item not in self.config.drop_output_types
                ],
            },
        }
        return metadata

    def _prepare_ground_truth_df(self, location: str) -> pd.DataFrame:
        """Filter and prepare ground truth observations for a location for ALL targets."""
        filtered = self.target_data[self.target_data["location"] == location].copy()

        if "target" not in filtered.columns:
            if self.config.ground_truth_value_key:
                filtered["target"] = self.config.ground_truth_value_key
            else:
                raise KeyError(
                    "A 'target' column is missing from the ground truth data, and no "
                    "'ground_truth_value_key' is configured to serve as a default."
                )

        if filtered.empty:
            return filtered

        date_col = self.config.ground_truth_date_column
        
        filtered["as_of"] = pd.to_datetime(filtered["as_of"])
        filtered[date_col] = pd.to_datetime(filtered[date_col])

        # --- CORRECTED REVISION LOGIC ---
        # 1. Sort by as_of date to get the most recent records last.
        filtered.sort_values("as_of", inplace=True)

        # 2. IMPORTANT: Drop rows where the observation is null. This prioritizes
        #    records with actual data during the deduplication step.
        filtered.dropna(subset=[self.config.observation_column], inplace=True)

        # 3. Now, drop duplicates, keeping the last (most recent) record that has a valid observation.
        filtered.drop_duplicates(subset=[date_col, "target"], keep="last", inplace=True)
        # --- END OF CORRECTIONS ---

        if self.config.ground_truth_min_date is not None:
            min_date = self.config.ground_truth_min_date
            if not isinstance(min_date, pd.Timestamp):
                min_date = pd.Timestamp(min_date)
            filtered = filtered[filtered[date_col] >= min_date]

        filtered.sort_values(date_col, inplace=True)
        return filtered

    def _format_ground_truth_output(self, ground_truth_df: pd.DataFrame) -> Dict[str, Any]:
        """Format ground truth DataFrame as a multi-target JSON-ready dictionary."""
        if ground_truth_df.empty:
            return {"dates": []}

        date_col = self.config.ground_truth_date_column
        pivot_truth = ground_truth_df.pivot(
            index=date_col, 
            columns="target", 
            values=self.config.observation_column
        )
        pivot_truth.sort_index(inplace=True)

        ground_truth = {
            "dates": pivot_truth.index.strftime('%Y-%m-%d').tolist()
        }
        for target_column in pivot_truth.columns:
            values_list = pivot_truth[target_column].tolist()
            ground_truth[target_column] = [None if pd.isna(v) else v for v in values_list]

        return ground_truth

    def _build_forecasts_key(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build the forecasts section of an individual JSON file."""
        forecasts: Dict[str, Any] = {}
        full_gbo = df.groupby(["reference_date", "target", "model_id", "horizon", "output_type"])
        for _, grouped_df in full_gbo:
            output_type = grouped_df["output_type"].iloc[0]
            if output_type in self.config.drop_output_types:
                continue

            reference_date = str(grouped_df["reference_date"].iloc[0])
            target = str(grouped_df["target"].iloc[0])
            model = str(grouped_df["model_id"].iloc[0])
            horizon = str(grouped_df["horizon"].iloc[0])

            reference_date_dict = forecasts.setdefault(reference_date, {})
            target_dict = reference_date_dict.setdefault(target, {})
            model_dict = target_dict.setdefault(model, {})

            if output_type == "quantile":
                model_dict["type"] = "quantile"
                predictions_dict = model_dict.setdefault("predictions", {})
                predictions_dict[horizon] = {
                    "date": str(grouped_df["target_end_date"].iloc[0]),
                    "quantiles": list(grouped_df["output_type_id"]),
                    "values": list(grouped_df["value"]),
                }
            elif output_type == "pmf":
                model_dict["type"] = "pmf"
                predictions_dict = model_dict.setdefault("predictions", {})
                predictions_dict[horizon] = {
                    "date": str(grouped_df["target_end_date"].iloc[0]),
                    "categories": list(grouped_df["output_type_id"]),
                    "probabilities": list(grouped_df["value"]),
                }
            else:
                raise ValueError(
                    "`output_type` of input data must either be 'quantile' or 'pmf', "
                    f"received '{output_type}'"
                )

        return forecasts

    def _build_available_models_list(self, df: pd.DataFrame) -> list:
        """Build list of models available for a specific location."""
        unique_models_from_loc_df = dict.fromkeys(df["model_id"])
        return [str(model) for model in unique_models_from_loc_df.keys()]

    def _build_all_models_list(self) -> list:
        """Build list of all models seen across the dataset."""
        unique_models_from_primary_df = dict.fromkeys(self.df_data["model_id"])
        return [str(model) for model in unique_models_from_primary_df.keys()]

    def _build_metadata_file(self, all_models: list[str]) -> Dict[str, Any]:
        """Build dataset-level metadata.json contents."""
        metadata_file_contents = {
            "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": sorted(all_models),
            "locations": [],
        }
        for _, row in self.locations_data.iterrows():
            location_info = {
                "location": str(row["location"]),
                "abbreviation": str(row["abbreviation"]),
                "location_name": str(row["location_name"]),
                "population": None if row["population"] is None else float(row["population"]),
            }
            metadata_file_contents["locations"].append(location_info)

        return metadata_file_contents
