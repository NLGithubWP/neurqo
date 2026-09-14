#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
pgdb_root=${PGDB_ROOT:-"$repo/../pgdb"}
runtime_root="$pgdb_root/.nqo_runtime/reproduction/job-data-scale"
sql_lock="$runtime_root/sql.lock"
policy_container=pgdb_tpch_gpu
policy_host=$(docker inspect "$policy_container" \
  --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
mkdir -p "$runtime_root/logs"

run_scale() {
  local scale=$1
  local base_port=$2
  local database="imdb_scale_${scale}"
  local run_root="$runtime_root/scale_${scale}"
  local cache="$run_root/experience.sql"
  local output_root=${NQO_OUTPUT_ROOT:-"$repo/results/benchmark/temp_nqo_reproduction"}
  local output="$output_root/nqo_job_scale_${scale}_runs.csv"
  mkdir -p "$run_root"
  mkdir -p "$output_root"
  if [[ ! -f "$cache" ]]; then
    cp "$repo/results/buffers/job_scale_${scale}.sql" "$cache"
  fi

  python3 "$repo/scripts/reproduce/nqo/run.py" \
    --dataset job --method postgres \
    --database "$database" --output "$output" \
    --runtime-dir "$run_root/postgres" \
    --sql-execution-lock "$sql_lock" --sql-execution-slots 3 \
    >"$runtime_root/logs/scale_${scale}_postgres.log" 2>&1

  local pids=()
  local folds=(a b c)
  local index=0
  for fold in "${folds[@]}"; do
    python3 "$repo/scripts/reproduce/nqo/run.py" \
      --dataset job --method nqo --protocol random --fold "$fold" \
      --database "$database" --cache "$cache" --output "$output" \
      --runtime-dir "$run_root/random_${fold}" \
      --container "$policy_container" \
      --server-listen-host 0.0.0.0 --server-action-host "$policy_host" \
      --server-port $((base_port + index)) --model-device cuda \
      --cache-miss execute \
      --sql-execution-lock "$sql_lock" --sql-execution-slots 3 \
      >"$runtime_root/logs/scale_${scale}_random_${fold}.log" 2>&1 &
    pids+=("$!")
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

scales=("$@")
if [[ ${#scales[@]} -eq 0 ]]; then
  scales=(50 25)
fi

pids=()
for scale in "${scales[@]}"; do
  run_scale "$scale" $((18200 + scale)) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
