#!/usr/bin/env bash

# Exit on error, unset vars, and show each command as it runs.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

repos=(
  "FluSight-forecast-hub|https://github.com/cdcepi/FluSight-forecast-hub.git"
  "rsv-forecast-hub|https://github.com/CDCgov/rsv-forecast-hub.git"
  "covid19-forecast-hub|https://github.com/CDCgov/covid19-forecast-hub.git"
  "flu-metrocast|https://github.com/reichlab/flu-metrocast.git"
)

for entry in "${repos[@]}"; do
  repo_dir="${entry%%|*}"
  repo_url="${entry##*|}"

  if [[ -d "${repo_dir}/.git" ]]; then
    git -C "${repo_dir}" pull --ff-only
  elif [[ -d "${repo_dir}" ]]; then
    echo "Directory ${repo_dir} exists but is not a git repository. Aborting." >&2
    exit 1
  else
    git clone "${repo_url}" "${repo_dir}"
  fi
done

mkdir -p app/public/processed_data

python scripts/process_RespiLens_data.py \
  --output-path "${SCRIPT_DIR}/app/public/processed_data" \
  --flusight-hub-path "${SCRIPT_DIR}/FluSight-forecast-hub" \
  --rsv-hub-path "${SCRIPT_DIR}/rsv-forecast-hub" \
  --covid-hub-path "${SCRIPT_DIR}/covid19-forecast-hub" \
  --flu-metrocast-hub-path "${SCRIPT_DIR}/flu-metrocast" \
  --NHSN \
  --NSSP
