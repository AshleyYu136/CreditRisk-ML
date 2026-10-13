# Methodology

## Data contract and quality

The original UCI spreadsheet has 30,000 rows, ID, 23 predictors and `default payment next month`, renamed `default`. The downloader verifies the original bytes with SHA-256, not a generated CSV's checksum. Required fields are numeric and finite; target must have both binary classes. Duplicate IDs, invalid ages/limits, negative payment amounts and unsupported category codes fail validation. Empty data fails. Outliers are counted using 1.5×IQR, not automatically deleted; large balances and payments can be valid financial observations. Negative bill amounts can reflect credit balances and are retained.

The observed file has no missing cells, 35 duplicate records excluding ID, 345 ambiguous education codes and 54 ambiguous marriage codes. Customer IDs are unique. Repeated original feature vectors, including potentially contradictory target labels, stay in a single split and CV fold. This avoids leakage without deleting genuine separate customer records. Training-only imputers are still present for robustness, while user inference requires complete validated inputs.

## Financial features

Every feature uses only the six observed historical months. ID and future default are excluded:

| Feature | Calculation | Interpretation and limit |
|---|---|---|
| bill_mean | Mean BILL_AMT1…6 | Statement balance level; not income or cash flow |
| payment_mean | Mean PAY_AMT1…6 | Historical payment level; not proof of affordability |
| utilization_proxy | BILL_AMT1 / max(LIMIT_BAL,1) | Statement balance relative to limit; not transaction-level utilization and may exceed 1 or be negative |
| payment_to_positive_bill | Sum payments / max(sum positive bills,1) | Historical repayment proxy; aggregated statement periods are imperfectly aligned, ratios may exceed 1 |
| max_delay | Max of repayment codes clipped below at zero | Worst positive reported delay; special negative codes are not negative months |
| delinquent_months | Count of repayment codes >0 | Number of reported delayed months |
| recent_bill_change | BILL_AMT1 − BILL_AMT6 | Recent versus earliest observed statement change; not a trend forecast |

Education 0/5/6 is mapped to documented `other` code 4; marriage 0 is mapped to `other` 3. Logistic regression treats categorical codes nominally using one-hot encoding; trees split on their numeric codes. Repayment negative codes distinguish source states, while positive codes are months delayed. Demographic numeric codes have no inherent metric ordering; tree partitions may impose an arbitrary ordering, a benchmark limitation. Continuous values are median-imputed; categorical values use most-frequent imputation. Scaling applies only to logistic regression. Imputation/statistics/category discovery stay inside the pipeline.

## Splits and search

Feature-vector groups receive a majority target label for stratifying group allocation (ties use default). Seed 42 creates approximate 70/15/15 training/validation/test partitions; slight row-count deviations preserve exact groups. The partition IDs are saved. Within training, StratifiedGroupKFold with three folds prevents duplicate vectors crossing folds. All feature engineering is deterministic and all learned preprocessing is refitted within each fold.

Five models: regularized logistic regression, depth/leaf-controlled decision tree, random forest, histogram XGBoost and LightGBM. Fast search draws two hyperparameter combinations per model; comprehensive search draws eight and increases ensemble size to 300. Search ranks by mean CV ROC-AUC. Baseline validation metrics are retained separately; tuning is not guaranteed to improve them. Train versus validation AUC documents overfitting. Boosters use fixed estimator counts rather than an external early-stopping set, ensuring fold-level isolation.

Logistic/LightGBM class-weight candidates and XGBoost scale_pos_weight candidates explicitly examine imbalance tradeoffs. Tree and random-forest leaf constraints reduce overfit; random forest uses parallel fitting. Search itself is serial to prevent nested worker oversubscription. There is no synthetic oversampling. Model parameters, random seed, library versions, dataset digest and transformed feature names are saved.

Only tuned validation ROC-AUC selects the winner. The separate best validation tree supports SHAP if the overall winner is logistic. Thresholds are selected on validation data before held-out test evaluation. Test metrics for every preregistered candidate are descriptive comparisons; they are not used to change the winner. No chronology/production drift/causal effect is claimed.

## Metrics and probability quality

ROC-AUC ranks positive above negative cases; AP integrates precision at recall increments and emphasizes minority-class retrieval. KS is max(TPR−FPR), with higher predicted scores indicating higher default risk. Brier is mean squared probability error. Confusion matrix is [[TN,FP],[FN,TP]], precision/recall/F1 use probability ≥0.5, zero-division results are zero. Calibration curves use quantile bins and compare average predicted probability with observed default fraction. Class weighting can distort probability calibration; raw probabilities are evaluated rather than assumed calibrated. A model would need a training-only calibration comparison before deployment.

## SQL

SQLAlchemy's DuckDB dialect creates an in-memory `clients` table from the observed dataset. Four versioned SQL queries check quality, analyze age/default/credit limit segments, rank repayment-status groups with DENSE_RANK, and calculate historical bill/payment averages with PERCENT_RANK and partitioned averages. No fictional transaction table or unnecessary joins are used. Tests compare SQL results to Pandas. Full training also validates SQL bill means before fitting. SQL summaries are limited to 30 rows per query; feature aggregation is computed for all customers.

## WOE/IV

`WOEAnalysis` fits training quantiles only, extends outer edges to infinity and uses an explicit missing bin if present. Good=nondefault and bad=default. Add 0.5 to good/bad bin counts; normalize each class distribution with that total pseudocount. WOE = log(good share / bad share). IV sums (good share − bad share)×WOE. Zero-count bins remain finite. Bins absent from fitted mappings use neutral WOE 0. The experiment reports credit-limit WOE/IV as a separate exploratory module, not a trained regulatory scorecard or production scoring system. IV measures association, not causation; conventional thresholds are not universal feature-selection rules. Nothing is fitted on validation/test labels.

## SHAP

TreeExplainer uses the exact pipeline-transformed inputs with tree-path-dependent background behavior. The pinned XGBoost 3.0.5 and SHAP 0.49.1 avoid the XGBoost 3.1 vector-intercept parser mismatch. Tree/forest class-1 SHAP is in probability units; boosted binary models use raw log-odds. Local additive sums are explicitly reconstructed and compared to predict_proba. Three-dimensional binary-class explanations select class 1. Mean absolute SHAP uses 150 validation customers, with a beeswarm, bar, dependence scatter and first-customer waterfall saved. Correlated inputs distribute credit according to the explainer's assumptions. SHAP is neither causal inference nor proof of why a person defaulted.

## Strategy and PSI

The hypothetical strategy maximizes validation approval fraction subject to observed approved bad rate ≤10% across a fixed 31-point approval grid, with review boundary 0.50. Both the target constraint and review boundary are configurable. If no candidate is feasible, a flagged fallback of 0.05 is reported; it is not labeled compliant. Ties select the larger approval boundary. Low is p<approval; medium is approval≤p<review; high is p≥review. The final test report uses these frozen choices. Empty approvals yield null approved default rate; absent defaults yield null capture rate. Rejection and review are simulated actions; neither realized profit nor ECL can be inferred without exposure/loss/cost and counterfactual assumptions.

PSI fits numerical quantile edges on training, adds infinite tails and an explicit missing bin. Categorical bins use the union of observed categories so new comparison categories retain mass. Half-count smoothing is normalized by sample size and number of bins. PSI = Σ(comparison−reference)×log(comparison/reference). Constant and all-missing reference samples are supported. PSI flags at 0.10/0.25 are contextual heuristics. Default comparison is a validation holdout, not production; dashboard shift is explicitly simulated. Changes can reflect sampling, quality issues or environment changes, not necessarily model degradation.
