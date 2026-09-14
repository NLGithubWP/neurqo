-- STACK (StackOverflow) database schema info
-- Database name: so
-- Source: pg_restore from so_pg12 dump (https://rmarcus.info/stack.html)
--
-- Tables (10):
--   account       ~13.9M rows
--   answer        ~6.4M rows
--   badge         ~51.2M rows
--   comment       ~103.5M rows
--   post_link     ~2.3M rows
--   question      ~12.7M rows
--   site          173 rows
--   so_user       ~21.1M rows
--   tag           ~186K rows
--   tag_question  ~36.9M rows

-- List all tables
\dt

-- Show schema for each table
\d account
\d answer
\d badge
\d comment
\d post_link
\d question
\d site
\d so_user
\d tag
\d tag_question

-- Row counts
SELECT relname, reltuples::bigint AS rows
FROM pg_class
WHERE relkind='r'
  AND relnamespace=(SELECT oid FROM pg_namespace WHERE nspname='public')
ORDER BY reltuples DESC;

-- List indexes
SELECT tablename, indexname FROM pg_indexes
WHERE schemaname='public'
ORDER BY tablename, indexname;

-- List foreign keys
SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_name;
