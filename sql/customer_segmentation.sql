WITH segments AS (
 SELECT *, CASE WHEN "AGE" < 30 THEN 'under_30' WHEN "AGE" < 50 THEN '30_to_49' ELSE '50_plus' END AS age_segment
 FROM clients
)
SELECT age_segment, COUNT(*) AS customers, AVG("default") AS default_rate,
AVG("LIMIT_BAL") AS mean_limit FROM segments GROUP BY age_segment ORDER BY age_segment;
