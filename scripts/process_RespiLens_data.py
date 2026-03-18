"""Pull/process all data required for RespiLens"""

import argparse
import logging
import pandas as pd
import sys
from pathlib import Path
from hubdata import connect_hub
from hubdata import connect_target_data
from hubdata.create_target_data_schema import TargetType


from processors import FlusightDataProcessor, RSVDataProcessor, COVIDDataProcessor, FluMetrocastDataProcessor
from nhsn_data_processor import NHSNDataProcessor
from helper import save_json_file, hubverse_df_preprocessor, clean_nan_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description = 'Pull/process all data required for RespiLens.')
    parser.add_argument("--output-path",
                        type=str,
                        required=False,
                        help="Absolute path where you want to save data do.")
    parser.add_argument("--flusight-hub-path",
                        type=str,
                        required=False,
                        help="Absolute path to local clone of FluSight forecast repo.")
    parser.add_argument("--rsv-hub-path",
                        type=str,
                        required=False,
                        help="Absolute path to local clone of RSV forecast repo.")
    parser.add_argument("--covid-hub-path",
                        type=str,
                        required=False,
                        help="Absolute path to local clone of COVID19 forecast repo.")
    parser.add_argument("--flu-metrocast-hub-path",
                        type=str,
                        required=False,
                        help="Absolute path to local clone of flu-metrocast repo.")
    parser.add_argument("--NHSN",
                        action='store_true',
                        required=False,
                        help="If set, pull NHSN data.") 
    args = parser.parse_args()

    if not (args.flusight_hub_path or args.rsv_hub_path or args.covid_hub_path or args.NHSN or args.flu_metrocast_hub_path):
        print("🛑 No hub paths or NHSN flag provided 🛑, so no data will be fetched.")
        print("Please re-run script with hub path(s) specified or NHSN flag set.")
        sys.exit(1)

    logger.info("Beginning conversion process...")

    if args.flusight_hub_path:
        # Use HubdataPy to get all Flusight data in one df
        logger.info("Establishing connection to local FluSight repository...")
        flu_hub_conn = connect_hub(args.flusight_hub_path)
        logger.info("Success ✅")
        logger.info("Collecting data from FluSight repo...")
        flu_hubverse_df = clean_nan_values(hubverse_df_preprocessor(df=flu_hub_conn.get_dataset().to_table().to_pandas(), filter_nowcasts=True))
        flu_locations_data = clean_nan_values(pd.read_csv(Path(args.flusight_hub_path) / 'auxiliary-data/locations.csv'))
        flu_target_data = clean_nan_values(connect_target_data(hub_path=args.flusight_hub_path, target_type=TargetType.TIME_SERIES).to_table().to_pandas()) 
        logger.info("Success ✅")
        # Initialize converter object
        flu_processor_object = FlusightDataProcessor(
            data=flu_hubverse_df,
            locations_data=flu_locations_data,
            target_data=flu_target_data,
        )
        # Iteratively save output files
        logger.info("Saving flu JSON files...")
        for filename, contents in flu_processor_object.output_dict.items():
            save_json_file(
                pathogen='flusight',
                output_path=args.output_path,
                output_filename=filename,
                file_contents=contents,
                overwrite=True
            )
        logger.info("Success ✅")

    
    if args.rsv_hub_path:
        # Use HubdataPy to get all RSV data in one df
        logger.info("Establishing connection to local RSV repository...")
        rsv_hub_conn = connect_hub(args.rsv_hub_path)
        logger.info("Success ✅")
        logger.info("Collecting data from RSV repo...")
        rsv_hubverse_df = clean_nan_values(hubverse_df_preprocessor(df=rsv_hub_conn.get_dataset().to_table().to_pandas(), filter_nowcasts=True))
        rsv_locations_data = clean_nan_values(pd.read_csv(Path(args.rsv_hub_path) / 'auxiliary-data/locations.csv'))
        rsv_target_data = clean_nan_values(connect_target_data(hub_path=args.rsv_hub_path, target_type=TargetType.TIME_SERIES).to_table().to_pandas()) 
        logger.info("Success ✅")
        # Initialize converter object
        rsv_processor_object = RSVDataProcessor(
            data=rsv_hubverse_df,
            locations_data=rsv_locations_data,
            target_data=rsv_target_data,
        )
        # Iteratively save output files
        logger.info("Saving RSV JSON files...")
        for filename, contents in rsv_processor_object.output_dict.items():
            save_json_file(
                pathogen='rsvforecasthub',
                output_path=args.output_path,
                output_filename=filename,
                file_contents=contents,
                overwrite=True
            )
        logger.info("Success ✅")
    
    if args.covid_hub_path:
        # Use HubdataPy to get all covid19 data in one df
        logger.info("Establishing connection to local covid19 repository...")
        covid_hub_conn = connect_hub(args.covid_hub_path)
        logger.info("Success ✅")
        logger.info("Collecting data from covid19 repo...")
        covid_hubverse_df = clean_nan_values(hubverse_df_preprocessor(df=covid_hub_conn.get_dataset().to_table().to_pandas(), filter_nowcasts=True))
        covid_locations_data = clean_nan_values(pd.read_csv(Path(args.covid_hub_path) / 'auxiliary-data/locations.csv'))
        covid_target_data = clean_nan_values(connect_target_data(hub_path=args.covid_hub_path, target_type=TargetType.TIME_SERIES).to_table().to_pandas())
        logger.info("Success ✅")
        # Initialize converter object
        covid_processor_object = COVIDDataProcessor(
            data=covid_hubverse_df,
            locations_data=covid_locations_data,
            target_data=covid_target_data,
        )
        # Iteratively save output files
        logger.info("Saving covid19 JSON files...")
        for filename, contents in covid_processor_object.output_dict.items():
            save_json_file(
                pathogen='covid19forecasthub',
                output_path=args.output_path,
                output_filename=filename,
                file_contents=contents,
                overwrite=True
            )
        logger.info("Success ✅")
    
    if args.flu_metrocast_hub_path:
        # Use HubdataPy to get all flu-metrocast data in one df
        logger.info("Establishing connection to local flu metrocast repository...")
        flu_metrocast_hub_conn = connect_hub(args.flu_metrocast_hub_path)
        logger.info("Success ✅")
        logger.info("Collecting data from flu metrocast repo...")
        flu_metrocast_hubverse_df = clean_nan_values(hubverse_df_preprocessor(df=flu_metrocast_hub_conn.get_dataset().to_table().to_pandas(), filter_nowcasts=True))
        flu_metrocast_locations_data = clean_nan_values(pd.read_csv(Path(args.flu_metrocast_hub_path) / 'auxiliary-data/locations.csv'))
        flu_metrocast_target_data = clean_nan_values(connect_target_data(hub_path=args.flu_metrocast_hub_path, target_type=TargetType.TIME_SERIES).to_table().to_pandas())
        logger.info("Success ✅")
        # Initialize converter object
        flu_metrocast_processor_object = FluMetrocastDataProcessor(
            data=flu_metrocast_hubverse_df,
            locations_data=flu_metrocast_locations_data,
            target_data=flu_metrocast_target_data
        )
        # Iteratively save output files
        logger.info("Saving flu metrocast JSON files...")
        for filename, contents in flu_metrocast_processor_object.output_dict.items():
            save_json_file(
                pathogen='flumetrocast',
                output_path=args.output_path,
                output_filename=filename,
                file_contents=contents,
                overwrite=True
            )
        logger.info("Success ✅")

    if args.NHSN:
        NHSN_processor_object = NHSNDataProcessor(resource_id='ua7e-t2fy', replace_column_names=True)
        logger.info("Iteratively saving NHSN JSON files...")
        for filename, contents in NHSN_processor_object.output_dict.items():
            save_json_file(
                pathogen="nhsn",
                output_path=args.output_path,
                output_filename=filename,
                file_contents=contents,
                overwrite=True
            )
        logger.info("Success ✅")
    
    logger.info("Process complete.")


if __name__ == "__main__":
    main()
    
