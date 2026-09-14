select count(*) from tag t, site s, question q, tag_question tq
where
s.site_name='raspberrypi' and
t.name='timekeeping' and
t.site_id = s.site_id and
q.site_id = s.site_id and
tq.site_id = s.site_id and
tq.question_id = q.id and
tq.tag_id = t.id
