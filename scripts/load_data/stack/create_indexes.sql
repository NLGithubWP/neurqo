-- STACK (StackOverflow) supplemental indexes for the `so` database.
--
-- Scope:
--   This file is not the complete index inventory for STACK.
--   It only contains additional indexes that were added on top of the
--   restored database to improve query performance for the current workload.
--
-- Background:
--   The restored `so` database already contains primary-key indexes and some
--   pre-existing single-column/composite indexes. The statements below add
--   more join/filter indexes that were found to matter for the STACK queries,
--   especially q2/q3-style cross-site/tag workloads.
--
-- Experimental note:
--   Applying these indexes changes the physical design of the database.
--   Baseline and split runtimes measured before and after this script are not
--   directly comparable. If the goal is reproducible evaluation, all variants
--   should be rerun on the same indexed database state.
--
-- Idempotence:
--   All statements use IF NOT EXISTS and are safe to rerun.
--
-- Usage:
--   docker exec pgdb_dev_opt /code/pgdb-dev/psql/bin/psql -h localhost -d so -f /path/to/create_indexes.sql
--   or: psql -h localhost -p 15432 -U pgdb -d so -f scripts/load_data/stack/create_indexes.sql

-- =========================================================
-- FK join indexes (join key columns not already leading in existing indexes)
-- =========================================================

-- so_user.account_id → account.id
-- Used by: q2, q3 (cross-site user queries via account)
CREATE INDEX IF NOT EXISTS so_user_account_id_idx ON so_user(account_id);

-- badge.user_id → so_user.id
-- Used by: q13, q14, q15, q16 (badge queries)
-- Note: badge_pkey is (site_id, user_id, name, date) — user_id not leading
CREATE INDEX IF NOT EXISTS badge_user_id_idx ON badge(user_id);

-- tag_question.tag_id → tag.id
-- Used by: all tag-based queries (q11-q16)
-- Note: tag_question_pkey is (site_id, question_id, tag_id) — tag_id not leading
CREATE INDEX IF NOT EXISTS tag_question_tag_id_idx ON tag_question(tag_id);

-- tag_question.question_id → question.id
-- Used by: all tag-question join queries
-- Note: existing idx is (site_id, tag_id, question_id) — question_id not leading
CREATE INDEX IF NOT EXISTS tag_question_question_id_idx ON tag_question(question_id);

-- answer.question_id → question.id
-- Used by: q12, q13, q14 (answer-question joins)
-- Note: existing idx is (site_id, question_id) — question_id not leading
CREATE INDEX IF NOT EXISTS answer_question_id_idx ON answer(question_id);

-- comment.post_id → question.id / answer.id
-- Used by: q5, q6, q8 (comment queries)
-- Note: existing idx is (site_id, post_id) — post_id not leading
CREATE INDEX IF NOT EXISTS comment_post_id_idx ON comment(post_id);

-- comment.user_id → so_user.id
-- Used by: q5, q6 (comment-user joins)
-- Note: existing idx is (site_id, user_id) — user_id not leading
CREATE INDEX IF NOT EXISTS comment_user_id_idx ON comment(user_id);

-- post_link.post_id_from → question.id
-- Used by: q8 (post link queries)
CREATE INDEX IF NOT EXISTS post_link_post_id_from_idx ON post_link(post_id_from);

-- post_link.post_id_to → question.id
-- Used by: q8 (post link queries)
CREATE INDEX IF NOT EXISTS post_link_post_id_to_idx ON post_link(post_id_to);

-- =========================================================
-- Filter indexes (high-frequency WHERE clause columns)
-- =========================================================

-- site.site_name — used in every STACK query: s.site_name in ('stackoverflow', ...)
CREATE INDEX IF NOT EXISTS site_site_name_idx ON site(site_name);

-- tag(site_id, name) — used in most queries: t.name in ('java', 'python', ...)
-- Composite with site_id for site-scoped tag lookups
CREATE INDEX IF NOT EXISTS tag_site_id_name_idx ON tag(site_id, name);

-- =========================================================
-- Verify
-- =========================================================
SELECT tablename, indexname
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
