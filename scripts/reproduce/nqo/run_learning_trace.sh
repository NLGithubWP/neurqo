#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec python3 "$repo_root/scripts/reproduce/nqo/run_learning_trace.py" \
  --workers 30 "$@"
