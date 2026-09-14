-- Instance-wide settings used by the reproduced JOB, STACK, and TPC-H runs.

ALTER SYSTEM SET max_parallel_workers_per_gather = '0';
ALTER SYSTEM SET max_parallel_workers = '0';
ALTER SYSTEM SET jit = 'off';
ALTER SYSTEM SET geqo = 'off';

ALTER SYSTEM SET checkpoint_timeout = '30min';
ALTER SYSTEM SET checkpoint_completion_target = '0.9';
ALTER SYSTEM SET max_wal_size = '32GB';
ALTER SYSTEM SET min_wal_size = '4GB';

ALTER SYSTEM SET shared_buffers = '32GB';
ALTER SYSTEM SET effective_cache_size = '96GB';
ALTER SYSTEM SET work_mem = '128MB';
ALTER SYSTEM SET autovacuum = 'off';

ALTER SYSTEM SET shared_preload_libraries = 'pg_hint_plan';

SELECT pg_reload_conf();
