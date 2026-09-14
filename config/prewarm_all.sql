-- Prewarm every table and index in the public schema of the current database.
CREATE EXTENSION IF NOT EXISTS pg_prewarm;

DO $$
DECLARE
    relation_name text;
BEGIN
    FOR relation_name IN
        SELECT format('%I.%I', schemaname, tablename)
        FROM pg_tables
        WHERE schemaname = 'public'
        UNION ALL
        SELECT format('%I.%I', schemaname, indexname)
        FROM pg_indexes
        WHERE schemaname = 'public'
    LOOP
        BEGIN
            PERFORM pg_prewarm(relation_name::regclass);
            RAISE NOTICE 'Prewarmed %', relation_name;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Could not prewarm %: %', relation_name, SQLERRM;
        END;
    END LOOP;
END;
$$;
