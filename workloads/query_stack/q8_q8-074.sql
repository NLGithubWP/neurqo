select count(distinct q1.id) from
site s, post_link pl, question q1, question q2, comment c1, comment c2,
tag t, tag_question tq1, tag_question tq2
where
s.site_name = 'german' and
pl.site_id = s.site_id and

pl.site_id = q1.site_id and
pl.post_id_from = q1.id and
pl.site_id = q2.site_id and
pl.post_id_to = q2.id and

c1.site_id = q1.site_id and
c1.post_id = q1.id and

c2.site_id = q2.site_id and
c2.post_id = q2.id and

c1.date > c2.date and

t.name in ('python', 'c++', 'r', 'python-3.x', 'wpf') and
t.id = tq1.tag_id and
t.site_id = tq1.site_id and
t.id = tq2.tag_id and
t.site_id = tq1.site_id and

t.site_id = pl.site_id and

tq1.site_id = q1.site_id and
tq1.question_id = q1.id and
tq2.site_id = q2.site_id and
tq2.question_id = q2.id;

