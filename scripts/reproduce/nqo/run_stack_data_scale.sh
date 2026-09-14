#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
pgdb_root=${PGDB_ROOT:-"$repo/../pgdb"}
runtime_root="$pgdb_root/.nqo_runtime/reproduction/stack-data-scale"
sql_lock="$runtime_root/sql.lock"
policy_container=pgdb_tpch_gpu
policy_host=$(docker inspect "$policy_container" \
  --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
database=so_scale_50
cache="$runtime_root/experience.sql"
output_root=${NQO_OUTPUT_ROOT:-"$repo/results/benchmark/temp_nqo_reproduction"}
output="$output_root/nqo_stack_scale_50_runs.csv"
mkdir -p "$runtime_root/logs"
mkdir -p "$output_root"

if [[ ! -f "$cache" ]]; then
  cp "$repo/results/buffers/stack_scale_50.sql" "$cache"
fi

python3 "$repo/scripts/reproduce/nqo/run.py" \
  --dataset stack --method postgres \
  --database "$database" --output "$output" \
  --runtime-dir "$runtime_root/postgres" \
  --sql-execution-lock "$sql_lock" --sql-execution-slots 3 \
  >"$runtime_root/logs/postgres.log" 2>&1

pids=()
index=0
for fold in a b c; do
  python3 "$repo/scripts/reproduce/nqo/run.py" \
    --dataset stack --method nqo --protocol random --fold "$fold" \
    --database "$database" --cache "$cache" --output "$output" \
    --runtime-dir "$runtime_root/random_${fold}" \
    --container "$policy_container" \
    --server-listen-host 0.0.0.0 --server-action-host "$policy_host" \
    --server-port $((18350 + index)) --model-device cuda \
    --cache-miss execute \
    --sql-execution-lock "$sql_lock" --sql-execution-slots 3 \
    >"$runtime_root/logs/random_${fold}.log" 2>&1 &
  pids+=("$!")
  index=$((index + 1))
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
