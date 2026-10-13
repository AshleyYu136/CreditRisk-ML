SELECT "PAY_0" AS latest_status, COUNT(*) AS customers,
SUM("default") AS defaults, AVG("default") AS default_rate,
AVG("LIMIT_BAL") AS mean_limit,
DENSE_RANK() OVER (ORDER BY AVG("default") DESC) AS risk_rank
FROM clients GROUP BY "PAY_0" ORDER BY risk_rank;
