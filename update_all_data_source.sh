# show on error and plot the commands
set -ex

echo "Processing FluSight data..."
python scripts/process_flusight_data.py  --hub-path FluSight-forecast-hub  --output-path app/public/processed_data

echo "Processing NHSN data..."
python scripts/process_nhsn_data.py --output-path app/public/processed_data --locations-path ./FluSight-forecast-hub/auxiliary-data/locations.csv

echo "Processing RSV data..."
python scripts/process_rsv_data.py  --hub-path rsv-forecast-hub  --output-path app/public/processed_data

echo "Running validation on processed data..."
python scripts/validate_processed_data.py app/public/processed_data
echo "Validation complete."

echo "All data processing and validation finished successfully."