-- TPC-H Query 18


select
        c.c_name,
        c.c_custkey,
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        sum(l.l_quantity)
from
        customer c,
        orders o,
        lineitem l
where
        o.o_orderkey in (
                select
                        l2.l_orderkey
                from
                        lineitem l2
                group by
                        l2.l_orderkey having
                                sum(l2.l_quantity) > 300
        )
        and c.c_custkey = o.o_custkey
        and o.o_orderkey = l.l_orderkey
group by
        c.c_name,
        c.c_custkey,
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice
order by
        o.o_totalprice desc,
        o.o_orderdate
limit 100
