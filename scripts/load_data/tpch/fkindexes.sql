-- Indexes on foreign key columns and commonly filtered date columns
CREATE INDEX idx_nation_regionkey ON NATION(N_REGIONKEY);
CREATE INDEX idx_supplier_nationkey ON SUPPLIER(S_NATIONKEY);
CREATE INDEX idx_customer_nationkey ON CUSTOMER(C_NATIONKEY);
CREATE INDEX idx_partsupp_partkey ON PARTSUPP(PS_PARTKEY);
CREATE INDEX idx_partsupp_suppkey ON PARTSUPP(PS_SUPPKEY);
CREATE INDEX idx_orders_custkey ON ORDERS(O_CUSTKEY);
CREATE INDEX idx_orders_orderdate ON ORDERS(O_ORDERDATE);
CREATE INDEX idx_lineitem_orderkey ON LINEITEM(L_ORDERKEY);
CREATE INDEX idx_lineitem_partkey ON LINEITEM(L_PARTKEY);
CREATE INDEX idx_lineitem_suppkey ON LINEITEM(L_SUPPKEY);
CREATE INDEX idx_lineitem_partkey_suppkey ON LINEITEM(L_PARTKEY, L_SUPPKEY);
CREATE INDEX idx_lineitem_shipdate ON LINEITEM(L_SHIPDATE);
CREATE INDEX idx_lineitem_commitdate ON LINEITEM(L_COMMITDATE);
CREATE INDEX idx_lineitem_receiptdate ON LINEITEM(L_RECEIPTDATE);
