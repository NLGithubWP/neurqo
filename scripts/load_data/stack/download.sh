#!/bin/bash
# Download the STACK (StackOverflow) PostgreSQL 12 dump
# Source: https://rmarcus.info/stack.html (Bao paper)
# File size: ~19GB

set -e

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
PGDB_ROOT=${PGDB_ROOT:-"$REPO_ROOT/../pgdb"}
DATA_DIR="${1:-$PGDB_ROOT/data/stack}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading STACK PG12 dump to $DATA_DIR ..."
wget -c 'https://www.dropbox.com/s/98u5ec6yb365913/so_pg12' -O so_pg12

echo "Download complete."
ls -lh so_pg12
