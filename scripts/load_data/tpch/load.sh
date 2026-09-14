#!/bin/bash
# Load TPC-H SF1 dataset into PostgreSQL inside Docker container.
#
# Prerequisites:
#   - Docker container 'pgdb_dev_opt' is running
#   - TPC-H .tbl files are at ../pgdb/data/tpch/
#     (copied from ../datasets/tpch_data/data_0/)
#
# The sibling ../pgdb repository is mounted as /code/pgdb-dev in the container.
#
# Usage:
#   bash load.sh                  # uses defaults
#   bash load.sh <db_name>        # custom database name (default: tpch)

set -e

CONTAINER="pgdb_dev_opt"
PSQL="/code/pgdb-dev/psql/bin/psql -h localhost"
DATA_DIR="/code/pgdb-dev/data/tpch"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DB_NAME="${1:-tpch}"

echo "=== TPC-H Data Loader ==="
echo "Container:  $CONTAINER"
echo "Database:   $DB_NAME"
echo "Data dir:   $DATA_DIR"
echo ""

# Step 1: Create database
echo "[1/5] Creating database '$DB_NAME'..."
docker exec "$CONTAINER" $PSQL -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || echo "Database '$DB_NAME' already exists, continuing..."

# Step 2: Create tables
echo "[2/5] Creating tables..."
docker exec "$CONTAINER" $PSQL -d "$DB_NAME" -f /code/pgdb-dev/data/tpch/schema.sql 2>/dev/null || \
  cat "$SCRIPT_DIR/schema.sql" | docker exec -i "$CONTAINER" $PSQL -d "$DB_NAME"

# Step 3: Load data (order matters for FK constraints)
echo "[3/5] Loading data..."
for t in nation region part supplier partsupp customer orders lineitem; do
  upper=$(echo "$t" | tr '[:lower:]' '[:upper:]')
  echo "  Loading $t..."
  docker exec "$CONTAINER" $PSQL -d "$DB_NAME" -c \
    "COPY $upper FROM '$DATA_DIR/$t.tbl' WITH (FORMAT csv, DELIMITER '|');"
done

# Step 4: Add constraints and indexes
echo "[4/5] Adding primary keys, foreign keys, and indexes..."
cat "$SCRIPT_DIR/add_constraints.sql" | docker exec -i "$CONTAINER" $PSQL -d "$DB_NAME"
cat "$SCRIPT_DIR/fkindexes.sql" | docker exec -i "$CONTAINER" $PSQL -d "$DB_NAME"

# Step 5: Analyze
echo "[5/5] Running ANALYZE..."
docker exec "$CONTAINER" $PSQL -d "$DB_NAME" -c "ANALYZE verbose;" 2>&1 | tail -5

# Verify
echo ""
echo "=== Verification ==="
docker exec "$CONTAINER" $PSQL -d "$DB_NAME" -c \
  "SELECT relname, reltuples::bigint AS rows FROM pg_class WHERE relkind='r' AND relnamespace=(SELECT oid FROM pg_namespace WHERE nspname='public') ORDER BY reltuples DESC;"

echo ""
echo "TPC-H dataset loaded successfully into database '$DB_NAME'."
