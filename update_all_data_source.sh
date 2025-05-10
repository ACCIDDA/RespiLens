# show on error and plot the commands
set -ex

Rscript scripts/process_nc_model_data.R

# python scripts/process_nc_data.py  --hub-path FluSight-forecast-hub  --output-path app/public/processed_data

python scripts/process_flusight_data.py  --hub-path FluSight-forecast-hub  --output-path app/public/processed_data

# python scripts/process_ncgt_data.py --output-path ./processed_data --locations-path ./FluSight-forecast-hub/auxiliary-data/locations.csv
python scripts/process_nhsn_data.py --output-path ./processed_data --locations-path ./FluSight-forecast-hub/auxiliary-data/locations.csv

# python scripts/process_rsv_data.py  --hub-path rsv-forecast-hub  --output-path app/public/processed_data 

