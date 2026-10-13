# Project interview guide

Answers describe the actual implementation and distinguish future work.

## 1. Why compare XGBoost with logistic regression?

Logistic regression supplies a regularized additive log-odds baseline and explicit coefficient contributions. XGBoost captures interactions and nonlinear effects through boosted trees. This implementation compares them on the same feature-group splits and uses validation AUC, not reputation, to choose the winner.

## 2. What is ROC-AUC?

It measures how often a randomly selected default case ranks above a nondefault case, with half-credit for tied scores. It assesses ranking across thresholds but does not guarantee useful precision, calibration or policy profitability.

## 3. How is KS computed here?

With higher scores representing greater default risk, compute the ROC curve and maximize TPR minus FPR. The perfect-ranking unit test gives 1, equal scores give 0 and reversed ranking gives 0 under this directional convention.

## 4. Why not choose by accuracy?

Nondefaults are 77.88% of the data, so an always-nondefault classifier looks accurate while retrieving no defaults. AUC, AP, recall, KS and probability-quality metrics are more informative.

## 5. What is leakage?

Information from evaluation samples influences fitted preprocessing, parameter selection, features or thresholds. This project excludes target/ID, fits preprocessing inside CV, groups repeated feature vectors, and freezes validation-based choices before test reporting.

## 6. Why group identical feature vectors?

The source has repeated records. Even distinct customer IDs can share the same original vector. Keeping each vector together avoids near-perfect duplicates straddling training and evaluation without deleting separate customers or choosing among conflicting labels.

## 7. Why a 70/15/15 design?

Training supports CV and parameter fitting, validation independently chooses models and policy boundaries, and test estimates frozen performance. Group stratification makes proportions approximate while preserving repeated-vector groups.

## 8. Was chronological validation performed?

No. This file contains six historical feature months but no reliable individual observation timeline for a future-cohort split. Random holdout results cannot be called out-of-time performance.

## 9. What does ColumnTransformer do?

It routes nominal codes through imputation and one-hot encoding for logistic regression and continuous features through median imputation and scaling. Tree models use unscaled numeric features. All learned operations are pipeline components.

## 10. Why no oversampling?

The dataset has about 22% defaults and reasonable sample size. Class weights are searched for selected models, while AUC/AP and Brier/calibration quantify tradeoffs. Oversampling is not automatically better and would need fold-local implementation.

## 11. How are hyperparameters tuned?

RandomizedSearchCV draws 2 configurations per model in fast mode or 8 in comprehensive mode, using 3 grouped-stratified training folds and ROC-AUC. CV details and best parameters are saved. Validation baseline and tuned scores expose regressions as well as improvements.

## 12. Why no booster early stopping?

Passing a shared external validation set to every CV fit could contaminate selection. This implementation uses fixed 80 or 300 tree budgets selected in advance. Fold-local early stopping could be added with carefully isolated inner holdouts.

## 13. What is the difference between AP and PR-AUC?

Average precision weights precision by successive recall increments, while trapezoidal PR-AUC linearly interpolates curve points. This project reports AP explicitly and does not claim the two are identical.

## 14. What does Brier score measure?

Mean squared error of probability forecasts against binary outcomes; lower is better. It reflects calibration and resolution. This implementation also displays quantile calibration curves rather than declaring probabilities calibrated from AUC alone.

## 15. How does SHAP work in this project?

TreeExplainer attributes the selected tree output among exact transformed model features under tree-path-dependent assumptions. Class 1 is chosen for multichannel outputs; additive totals are checked against predicted probability, using sigmoid for boosted log-odds.

## 16. Are SHAP values probabilities?

For the decision tree and random forest they are class-1 probability contributions. For XGBoost and LightGBM they are raw log-odds contributions. The UI labels scale and does not present a log-odds effect as a probability-point change.

## 17. What is global versus local importance?

Global mean absolute SHAP averages magnitudes over 150 validation customers; it summarizes the sampled model behavior. A local waterfall shows signed contributions for one customer. Neither is a causal explanation of default.

## 18. What are WOE and IV?

WOE is log(good distribution share / bad distribution share) within a bin. IV sums share differences times WOE. Training-only quantile bins and 0.5 pseudocount smoothing avoid leakage and infinite values. The module is exploratory, not a regulatory scorecard.

## 19. How are zero-count bins handled?

WOE adds 0.5 to each class count and renormalizes; PSI adds 0.5 to each sample-bin count and renormalizes. The implementation uses infinite outer numeric edges, explicit missing bins and safe neutral WOE for unmapped categories.

## 20. What is PSI?

Population Stability Index compares reference and comparison bin shares using sum((comparison-reference)*log(comparison/reference)). Numeric edges come from training only. New categorical values retain probability mass through the union of categories.

## 21. What drift is shown?

The default is a training-versus-validation comparison, not observed production drift. A dashboard checkbox multiplies comparison credit limits by 1.5 and clearly labels this synthetic shift. PSI flags are heuristics, not proof of performance degradation.

## 22. How are thresholds selected?

On validation predictions, a fixed grid maximizes approvals under an illustrative observed approved-default constraint of 10%; the review boundary defaults to 0.50. Frozen thresholds apply to test. Interactive sliders recalculate validation only.

## 23. What happens at a decision boundary?

Approval uses strict p<approval. Equality at approval is medium risk; equality at review is high risk. Tests explicitly check both boundaries. Empty approvals report a null observed default rate rather than a false zero.

## 24. Can strategy metrics estimate profit or ECL?

No. The source lacks exposure assumptions, LGD, recovery, interest and operational costs, as well as counterfactual outcomes under alternative approval policies. Reported rates are retrospective historical-label simulations.

## 25. What SQL skills are demonstrated?

DuckDB via SQLAlchemy executes GROUP BY segment rates, quality counts, CTE feature aggregation, DENSE_RANK risk ranking, PERCENT_RANK credit-limit ranking and partitioned outcome averages. Tests compare those outputs with Pandas instead of adding fictional tables.

## 26. What are the biggest dataset limitations?

It is one 2005 Taiwan cardholder portfolio, not current loan applicants. There is no observed income, transaction-level ledger, trustworthy temporal cohort, profit outcome or loss-given-default. Existing-customer selection and historical policy constrain external validity.

## 27. How could performance deteriorate in production?

Customer mix, economic conditions, source-code semantics, data quality or relationships between predictors and default can change. Monitor schema and distributions, and evaluate calibration/discrimination with delayed labels rather than relying on PSI alone.

## 28. What fairness work remains?

Sex, age, education and marriage are included for a benchmark but require legal and ethical feature review. Subgroup error rates, calibration, coverage and uncertainty have not been audited. No fairness certification is claimed.

## 29. Where would Spark or Hive fit?

At larger scale, Spark could validate and aggregate historical records in a data lake and Hive could expose partitioned analytical tables. Preserve point-in-time correctness and group/cohort splitting before extracting modeling samples. It is unnecessary for this 30,000-row file.

## 30. How would this become a production architecture?

Separate scheduled training from online inference, introduce signed/versioned artifacts, an authenticated prediction service with schema contracts, audit events, monitoring and delayed-label evaluation, and a governed rollback process. These are future improvements, not implemented deployment claims.

## 31. How are artifacts handled securely?

Only caller-explicitly trusted local Joblib pipelines are loaded. No model upload UI exists. File paths are fixed by project configuration. Hashes protect dataset provenance; a real registry would require model signing and a stronger deserialization trust boundary.

## 32. What does CI prove?

It checks deterministic fixture behavior, all model construction/prediction, SQL equivalence, SHAP additivity, metrics, strategy/PSI and missing-artifact UI behavior. It does not prove full-data generalization, Docker deployability or contemporary fairness. Full-data integration results are separately documented.
