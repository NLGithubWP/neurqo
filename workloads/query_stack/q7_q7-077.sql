select count(distinct acc.display_name) from account acc, so_user u, badge b1, badge b2 where
acc.website_url != '' and
acc.id = u.account_id and

b1.site_id = u.site_id and
b1.user_id = u.id and
b1.name = 'Legendary' and

b2.site_id = u.site_id and
b2.user_id = u.id and
b2.name = 'Documentation Beta' and
b2.date > b1.date + '6 months'::interval
