\set ON_ERROR_STOP on
\timing on

BEGIN;
SET LOCAL session_replication_role = replica;

CREATE TABLE __nqo_keep_question AS
SELECT site_id, id
FROM question
WHERE ((
    hashtextextended(site_id::text || ':' || id::text, 20260807)
    & 9223372036854775807::bigint
) % 100) < :keep_percent;

ALTER TABLE __nqo_keep_question ADD PRIMARY KEY (site_id, id);
ANALYZE __nqo_keep_question;

CREATE TABLE __nqo_keep_post (
    site_id integer NOT NULL,
    id integer NOT NULL,
    PRIMARY KEY (site_id, id)
);

INSERT INTO __nqo_keep_post
SELECT site_id, id FROM __nqo_keep_question;

INSERT INTO __nqo_keep_post
SELECT answer.site_id, answer.id
FROM answer
JOIN __nqo_keep_question AS keep
  ON keep.site_id = answer.site_id
 AND keep.id = answer.question_id
ON CONFLICT DO NOTHING;

ANALYZE __nqo_keep_post;

CREATE TABLE __nqo_all_post (
    site_id integer NOT NULL,
    id integer NOT NULL,
    PRIMARY KEY (site_id, id)
);

INSERT INTO __nqo_all_post
SELECT site_id, id FROM question;

INSERT INTO __nqo_all_post
SELECT site_id, id FROM answer
ON CONFLICT DO NOTHING;

ANALYZE __nqo_all_post;

DELETE FROM comment AS row
WHERE NOT EXISTS (
    SELECT 1
    FROM __nqo_keep_post AS keep
    WHERE keep.site_id = row.site_id
      AND keep.id = row.post_id
)
AND (
    EXISTS (
        SELECT 1
        FROM __nqo_all_post AS post
        WHERE post.site_id = row.site_id
          AND post.id = row.post_id
    )
    OR ((
        hashtextextended(row.site_id::text || ':' || row.id::text, 20260807)
        & 9223372036854775807::bigint
    ) % 100) >= :keep_percent
);

DELETE FROM tag_question AS row
WHERE NOT EXISTS (
    SELECT 1
    FROM __nqo_keep_question AS keep
    WHERE keep.site_id = row.site_id
      AND keep.id = row.question_id
);

DELETE FROM post_link AS row
WHERE NOT EXISTS (
          SELECT 1
          FROM __nqo_keep_question AS keep
          WHERE keep.site_id = row.site_id
            AND keep.id = row.post_id_from
      )
   OR NOT EXISTS (
          SELECT 1
          FROM __nqo_keep_question AS keep
          WHERE keep.site_id = row.site_id
            AND keep.id = row.post_id_to
      );

DELETE FROM answer AS row
WHERE NOT EXISTS (
    SELECT 1
    FROM __nqo_keep_question AS keep
    WHERE keep.site_id = row.site_id
      AND keep.id = row.question_id
);

DELETE FROM question AS row
WHERE NOT EXISTS (
    SELECT 1
    FROM __nqo_keep_question AS keep
    WHERE keep.site_id = row.site_id
      AND keep.id = row.id
);

DROP TABLE __nqo_all_post;
DROP TABLE __nqo_keep_post;
DROP TABLE __nqo_keep_question;
COMMIT;

SET maintenance_work_mem = '2GB';
VACUUM (FULL, ANALYZE) comment;
VACUUM (FULL, ANALYZE) tag_question;
VACUUM (FULL, ANALYZE) post_link;
VACUUM (FULL, ANALYZE) answer;
VACUUM (FULL, ANALYZE) question;
