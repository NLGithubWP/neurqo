# NeurQO Naming and Existing Artifacts

The system name is **NeurQO**. New commands, settings, environment variables,
and C identifiers use `neurqo`, `NEURQO_*`, and `NeurQO*` as appropriate.

- Server: `neurqo-server`; training tools: `neurqo-train`,
  `neurqo-train-mixed`, and `neurqo-incremental-trainer`.
- Research runner: `neurqo-benchmark` or `scripts/neurqo_benchmark.py`.
- Reproduction scripts: `scripts/reproduce/neurqo/`.
- PostgreSQL: `SET neurqo = on`, `neurqo.server_url`, and other `neurqo.*` GUCs.
- Kernel patch: `db_integration/neurqo-pg.patch`; its README specifies the base.

## Compatibility

Existing logs, SQLite buffers, checkpoints, and result CSVs are not renamed or
rewritten. Paths under `results/benchmark/nqo/` and filenames such as
`nqo_runs.csv` remain the released artifact locations.

- The server reads `NQO_*` environment variables when the corresponding
  `NEURQO_*` variable is absent. The new spelling takes precedence. The old
  `--nqo-src` option is accepted as an alias for `--neurqo-src`.
- PostgreSQL resolves old `nqo` and `nqo.*` settings to the same new GUCs.
  Existing database defaults and configuration files continue to work.
- Model module paths, network architecture, tensor names, and action-space
  dimensions are unchanged. The existing checkpoint loader still translates
  older action-head names in memory.
- Profile readers accept `nqo_enabled`. Cache lookup includes the old profile
  hashes. Persisted temporary-relation tokens and collector configuration
  identity retain their old spellings so renaming does not invalidate labels.
- Updated result readers translate `NQO` method labels and `nqo_*` CSV columns
  in memory. SQL text, model paths, and saved numerical measurements are not
  substituted. Analysis programs keep their existing filenames alongside the
  archived artifacts, but accept both spellings and display NeurQO.
- If a dataset buffer exists only under `.nqo_runtime/experience/`, the shared
  buffer resolver uses it directly instead of silently starting an empty one.

After rebuilding the kernel, install matching server headers and rebuild
`pg_hint_plan` and `pg_lip_bloom` against that installation. Existing binaries
from a different NeurDB node layout must not be reused.
