-- TPC-H Query 22


select
        cntrycode,
        count(*) as numcust,
        sum(c_acctbal) as totacctbal
from
        (
                select
                        substring(c.c_phone from 1 for 2) as cntrycode,
                        c.c_acctbal
                from
                        customer c
                where
                        substring(c.c_phone from 1 for 2) in
                                ('13', '31', '23', '29', '30', '18', '17')
                        and c.c_acctbal > (
                                select
                                        avg(c2.c_acctbal)
                                from
                                        customer c2
                                where
                                        c2.c_acctbal > 0.00
                                        and substring(c2.c_phone from 1 for 2) in
                                                ('13', '31', '23', '29', '30', '18', '17')
                        )
                        and not exists (
                                select
                                        *
                                from
                                        orders o
                                where
                                        o.o_custkey = c.c_custkey
                        )
        ) as custsale
group by
        cntrycode
order by
        cntrycode
