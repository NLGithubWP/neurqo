#!/bin/bash
# Load STACK (StackOverflow) dataset into PostgreSQL inside Docker container.
#
# Prerequisites:
#   - Docker container 'pgdb_dev_opt' is running
#   - The dump file has been downloaded via download.sh
#     (default location: ../pgdb/data/stack/so_pg12)
#
# The sibling ../pgdb repository is mounted as /code/pgdb-dev in the container.
#
# Usage:
#   bash load.sh                  # uses defaults
#   bash load.sh <db_name>        # custom database name (default: so)

set -e

CONTAINER="pgdb_dev_opt"
PSQL="/code/pgdb-dev/psql/bin/psql"
PG_RESTORE="/code/pgdb-dev/psql/bin/pg_restore"
DUMP_PATH="/code/pgdb-dev/data/stack/so_pg12"
DB_NAME="${1:-so}"

echo "=== STACK Data Loader ==="
echo "Container:  $CONTAINER"
echo "Database:   $DB_NAME"
echo "Dump file:  $DUMP_PATH"
echo ""

# Step 1: Verify dump file exists inside container
echo "[1/4] Checking dump file..."
docker exec "$CONTAINER" ls -lh "$DUMP_PATH"

# Step 2: Create database (ignore error if already exists)
echo "[2/4] Creating database '$DB_NAME'..."
docker exec "$CONTAINER" $PSQL -h localhost -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || echo "Database '$DB_NAME' already exists, continuing..."

# Step 3: Restore dump (~2.5 hours)
echo "[3/4] Restoring dump (this takes ~2.5 hours)..."
docker exec "$CONTAINER" $PG_RESTORE -h localhost -d "$DB_NAME" -j 4 --no-privileges --no-owner --verbose "$DUMP_PATH" 2>&1 | tail -10

# Step 4: Analyze
echo "[4/4] Running ANALYZE..."
docker exec "$CONTAINER" $PSQL -h localhost -d "$DB_NAME" -c "ANALYZE verbose;" 2>&1 | tail -5

# Verify
echo ""
echo "=== Verification ==="
docker exec "$CONTAINER" $PSQL -h localhost -d "$DB_NAME" -c \
  "SELECT relname, reltuples::bigint AS rows FROM pg_class WHERE relkind='r' AND relnamespace=(SELECT oid FROM pg_namespace WHERE nspname='public') ORDER BY reltuples DESC;"

echo ""
echo "STACK dataset loaded successfully into database '$DB_NAME'."
