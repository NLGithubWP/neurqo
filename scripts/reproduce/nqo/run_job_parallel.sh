#!/usr/bin/env bash
set -euo pipefail

cache_miss=${1:-error}
if [[ "$cache_miss" != "error" && "$cache_miss" != "execute" ]]; then
  echo "usage: $0 [error|execute]" >&2
  exit 2
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
pgdb_root=${PGDB_ROOT:-"$repo/../pgdb"}
output=${NQO_OUTPUT:-"$repo/results/benchmark/temp_nqo_reproduction/nqo_runs.csv"}
log_root="$pgdb_root/.nqo_runtime/reproduction/nqo-evaluator/logs"
mkdir -p "$log_root" "$(dirname "$output")"
cache_args=()
if [[ "$cache_miss" == "execute" ]]; then
  writable_cache="$pgdb_root/.nqo_runtime/reproduction/nqo-evaluator/job-experience.sql"
  if [[ ! -f "$writable_cache" ]]; then
    mkdir -p "$(dirname "$writable_cache")"
    cp "$repo/results/buffers/job_light.sql" "$writable_cache"
  fi
  cache_args=(--cache "$writable_cache")
fi

protocols=(base_query leave_one_out random)
folds=(a b c)
pids=()
labels=()
port=18095

python3 "$repo/scripts/reproduce/nqo/run.py" \
  --dataset job --method postgres --output "$output" \
  >"$log_root/postgres.log" 2>&1

cleanup() {
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for protocol in "${protocols[@]}"; do
  for fold in "${folds[@]}"; do
    label="${protocol}-${fold}"
    log="$log_root/${label}.log"
    (
      cd "$repo"
      exec python3 "$repo/scripts/reproduce/nqo/run.py" \
        --dataset job \
        --protocol "$protocol" \
        --fold "$fold" \
        --cache-miss "$cache_miss" \
        "${cache_args[@]}" \
        --server-port "$port" \
        --output "$output"
    ) >"$log" 2>&1 &
    pids+=("$!")
    labels+=("$label")
    echo "started $label pid=$! port=$port log=$log"
    port=$((port + 1))
  done
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed ${labels[$index]}"
  else
    status=$?
    echo "failed ${labels[$index]} status=$status" >&2
    failed=1
  fi
done

exit "$failed"
