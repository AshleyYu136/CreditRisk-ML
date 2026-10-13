WITH summary AS (
 SELECT "ID", "default", "LIMIT_BAL",
 ("BILL_AMT1"+"BILL_AMT2"+"BILL_AMT3"+"BILL_AMT4"+"BILL_AMT5"+"BILL_AMT6")/6.0 AS bill_mean,
 ("PAY_AMT1"+"PAY_AMT2"+"PAY_AMT3"+"PAY_AMT4"+"PAY_AMT5"+"PAY_AMT6")/6.0 AS payment_mean FROM clients
)
SELECT *, PERCENT_RANK() OVER (ORDER BY "LIMIT_BAL") AS limit_percentile,
AVG("LIMIT_BAL") OVER (PARTITION BY "default") AS mean_limit_by_outcome FROM summary;
