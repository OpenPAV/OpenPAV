#!/usr/bin/env bash
set -euo pipefail

run() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: $*"
  "$@"
}

# run python run_all.py --data_csv data/MicroSimACC/step3.csv --fav_id 0
run python run_all.py --data_csv data/Vanderbilt/step3_two_vehicle_ACC.csv --fav_id 0
run python run_all.py --data_csv data/Ohio/step3_single_vehicle.csv --fav_id 0
run python run_all.py --data_csv data/CATS/step3_ACC.csv --fav_id 0
# run python run_all.py --data_csv data/OpenACC/step3_Casale.csv --fav_id 0
# run python run_all.py --data_csv data/OpenACC/step3_Vicolungo.csv --fav_id 0
# run python run_all.py --data_csv data/OpenACC/step3_ASta.csv --fav_id 2
# run python run_all.py --data_csv data/OpenACC/step3_ZalaZone.csv --fav_id 4
# run python run_all.py --benchmark_model idm

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All runs completed."
