#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 /path/to/neurdb > neurqo-pg.patch" >&2
    exit 2
fi

# Upstream dev merged into neurqo-dev; excludes unrelated upstream changes.
base=cc0894372997d414fb36b9b1525df57c31d457b7
index_dir=$(mktemp -d)
trap 'rm -rf "$index_dir"' EXIT
export GIT_INDEX_FILE="$index_dir/index"
git -C "$1" read-tree "$base"
# A private index includes newly named files without staging the user's work.
git -C "$1" add -A -- \
    dbengine/src/backend/executor/Makefile \
    dbengine/src/backend/executor/nodeHashjoin.c \
    dbengine/src/backend/executor/nodeNeurQOAdaptiveJoin.c \
    dbengine/src/backend/parser/Makefile \
    dbengine/src/backend/parser/query_split.c \
    dbengine/src/backend/tcop/postgres.c \
    dbengine/src/backend/utils/misc/guc.c \
    dbengine/src/backend/utils/misc/guc_tables.c \
    dbengine/src/include/executor/nodeHashjoin.h \
    dbengine/src/include/executor/nodeNeurQOAdaptiveJoin.h \
    dbengine/src/include/parser/query_split.h
git -C "$1" diff --cached --no-ext-diff --no-textconv --binary --full-index "$base"
