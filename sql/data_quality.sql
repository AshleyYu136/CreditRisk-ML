SELECT COUNT(*) AS rows, COUNT(DISTINCT "ID") AS distinct_ids,
SUM(CASE WHEN "default" NOT IN (0,1) THEN 1 ELSE 0 END) AS invalid_targets,
SUM(CASE WHEN "LIMIT_BAL" <= 0 THEN 1 ELSE 0 END) AS invalid_limits,
SUM(CASE WHEN "AGE" IS NULL OR "default" IS NULL THEN 1 ELSE 0 END) AS missing_key_fields
FROM clients;
