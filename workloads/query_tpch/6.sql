-- TPC-H Query 6

select
        sum(l.l_extendedprice * l.l_discount) as revenue
from
        lineitem l
where
        l.l_shipdate >= date '1994-01-01'
        and l.l_shipdate < date '1995-01-01'
        and l.l_discount between 0.06 - 0.01 and 0.06 + 0.01
        and l.l_quantity < 24
