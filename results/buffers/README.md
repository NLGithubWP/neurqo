# Released execution buffers

These files are read-only SQLite databases containing execution experience
collected on the authors' PostgreSQL installation. They let NQO replay matching
state--action trajectories and train policies without executing the same SQL
again. Fresh experiments should write to a separate buffer path.

Despite the historical `.sql` suffix, each file is a SQLite database with one
table, `replay_cache`.

| Buffer | Size | Records | Queries | Purpose |
|---|---:|---:|---:|---|
| `job_light.sql` | 79.4 MiB | 11,045 | 113 | Main JOB experiments |
| `stack_light.sql` | 85.6 MiB | 12,090 | 112 | Main STACK experiments |
| `tpch_light.sql` | 0.7 MiB | 266 | 22 | Main TPC-H experiments |
| `job_scale_25.sql` | 0.8 MiB | 113 | 113 | JOB 25% data-scale experiment |
| `job_scale_50.sql` | 0.8 MiB | 113 | 113 | JOB 50% data-scale experiment |
| `job_scale_75.sql` | 0.8 MiB | 113 | 113 | JOB 75% data-scale experiment |
| `stack_scale_50.sql` | 0.9 MiB | 112 | 112 | STACK 50% data-scale experiment |

The released buffers total approximately 169 MiB.

## `replay_cache` schema

| Column | Meaning |
|---|---|
| `cache_id` | Stable identifier for the complete execution record; primary key. |
| `query_id` | Workload query identifier. |
| `sql_hash` | SHA-256 digest of the executed SQL text. |
| `trajectory_hash` | Digest of the ordered, semantically relevant policy states and actions. |
| `created_at_ms` | Record creation time in Unix milliseconds. |
| `action_config_hash` | Digest identifying the fixed action configuration used for execution. |
| `status` | Terminal execution status, such as `ok` or `timeout`. |
| `first_runtime_ms` | First end-to-end client-observed runtime in milliseconds. |
| `charged_runtime_ms` | Runtime charged to evaluation after applying timeout accounting. |
| `timeout_limit_ms` | Statement timeout used for the execution. |
| `result_hash` | Optional digest of the query result used for correctness checks. |
| `result_rows` | Optional result row count. |
| `source_episode_id` | Identifier of the source query-processing episode. |
| `trajectory_blob_hash` | SHA-256 digest of the uncompressed trajectory payload. |
| `trajectory_encoding` | Encoding of `trajectory_payload`; released records use zlib-compressed JSON. |
| `trajectory_payload` | Ordered policy state/action trajectory. |
| `db_events_blob_hash` | SHA-256 digest of the uncompressed database-event payload. |
| `db_events_encoding` | Encoding of `db_events_payload`; released records use zlib-compressed JSON. |
| `db_events_payload` | Database-side execution, timing, and materialization events. |
| `bindings_json` | Optional protocol, fold, and checkpoint-SHA bindings for exact replay. |

Experiment manifests, workload splits, checkpoints, and PostgreSQL baselines
are intentionally stored outside the buffer. The canonical implementation of
this format is [`src/experience/store.py`](../../src/experience/store.py).

The released payloads retain their original action fields. The buffer reader
maps those fields to Dec, Sched, Enum, Filter, and AJoin in memory. This legacy
adapter is read-only: released files are never rewritten, and newly appended or
merged records are stored only with the canonical action vocabulary.
