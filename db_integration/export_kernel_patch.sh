#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 /path/to/neurdb > nqo-pg.patch" >&2
    exit 2
fi

# Last NeurDB commit before the first NQO kernel integration.
base=421b2fab467afcab13845a845768cf73c416f5f8
git -C "$1" diff --no-ext-diff --no-textconv --binary --full-index "$base" -- \
    dbengine/src/backend/executor/Makefile \
    dbengine/src/backend/executor/nodeHashjoin.c \
    dbengine/src/backend/executor/nodeNqoAdaptiveJoin.c \
    dbengine/src/backend/parser/Makefile \
    dbengine/src/backend/parser/query_split.c \
    dbengine/src/backend/tcop/postgres.c \
    dbengine/src/backend/utils/misc/guc_tables.c \
    dbengine/src/include/executor/nodeHashjoin.h \
    dbengine/src/include/executor/nodeNqoAdaptiveJoin.h \
    dbengine/src/include/parser/query_split.h
