\set ON_ERROR_STOP on
\timing on

BEGIN;
SET LOCAL session_replication_role = replica;

CREATE TABLE __nqo_keep_title AS
SELECT id
FROM title
WHERE ((hashint4(id)::bigint & 2147483647::bigint) % 100)
      < :keep_percent;

ALTER TABLE __nqo_keep_title ADD PRIMARY KEY (id);
ANALYZE __nqo_keep_title;

DELETE FROM aka_title AS row
WHERE NOT EXISTS (
    SELECT 1 FROM __nqo_keep_title AS keep WHERE keep.id = row.movie_id
);

DELETE FROM cast_info AS row
WHERE NOT EXISTS (
    SELECT 1 FROM __nqo_keep_title AS keep WHERE keep.id = row.movie_id
);

DELETE FROM complete_cast AS row
WHERE NOT EXISTS (
    SELECT 1 FROM __nqo_keep_title AS keep WHERE keep.id = row.movie_id
);

DELETE FROM movie_companies AS row
WHERE NOT EXISTS (
    SELECT 1 FROM __nqo_keep_title AS keep WHERE keep.id = row.movie_id
);

DELETE FROM movie_info AS row
WHERE NOT EXISTS (
    SELECT 1 FROM __nqo_keep_title AS keep WHERE keep.id = row.movie_id
);

DELETE FROM movie_info_idx AS row
WHERE NOT EXISTS (
    SELECT 1 FROM __nqo_keep_title AS keep WHERE keep.id = row.movie_id
);

DELETE FROM movie_keyword AS row
WHERE NOT EXISTS (
    SELECT 1 FROM __nqo_keep_title AS keep WHERE keep.id = row.movie_id
);

DELETE FROM movie_link AS row
WHERE NOT EXISTS (
          SELECT 1 FROM __nqo_keep_title AS keep
          WHERE keep.id = row.movie_id
      )
   OR NOT EXISTS (
          SELECT 1 FROM __nqo_keep_title AS keep
          WHERE keep.id = row.linked_movie_id
      );

DELETE FROM title AS row
WHERE NOT EXISTS (
    SELECT 1 FROM __nqo_keep_title AS keep WHERE keep.id = row.id
);

DROP TABLE __nqo_keep_title;
COMMIT;

SET maintenance_work_mem = '2GB';
VACUUM (FULL, ANALYZE) aka_title;
VACUUM (FULL, ANALYZE) cast_info;
VACUUM (FULL, ANALYZE) complete_cast;
VACUUM (FULL, ANALYZE) movie_companies;
VACUUM (FULL, ANALYZE) movie_info;
VACUUM (FULL, ANALYZE) movie_info_idx;
VACUUM (FULL, ANALYZE) movie_keyword;
VACUUM (FULL, ANALYZE) movie_link;
VACUUM (FULL, ANALYZE) title;
