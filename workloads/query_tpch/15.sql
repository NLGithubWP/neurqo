-- TPC-H Query 15


with revenue as (
	select
		l.l_suppkey as supplier_no,
		sum(l.l_extendedprice * (1 - l.l_discount)) as total_revenue
	from
		lineitem l
	where
		l.l_shipdate >= date '1996-01-01'
		and l.l_shipdate < date '1996-04-01'
	group by
		l.l_suppkey)
select
	s.s_suppkey,
	s.s_name,
	s.s_address,
	s.s_phone,
	r.total_revenue
from
	supplier s,
	revenue r
where
	s.s_suppkey = supplier_no
	and r.total_revenue = (
		select
			max(r2.total_revenue)
		from
			revenue r2
	)
order by
	s.s_suppkey
