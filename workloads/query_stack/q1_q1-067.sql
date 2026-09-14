select count(*) from tag t, site s, question q, tag_question tq
where
s.site_name='stackoverflow' and
t.name='vb6' and
t.site_id = s.site_id and
q.site_id = s.site_id and
tq.site_id = s.site_id and
tq.question_id = q.id and
tq.tag_id = t.id
