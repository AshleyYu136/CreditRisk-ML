# Model card

## Intended use and exclusions

Academic portfolio: compare structured-data credit card default classifiers, financial risk metrics and hypothetical historical-label strategy simulations. It predicts next-month payment default for the UCI population. It is not approved for loan underwriting, pricing, automated denial of services, individual credit advice or regulated capital calculations. It is not a validated PD/EAD/LGD system. Dashboard labels are educational actions, not recommendations to a lender.

## Data

30,000 Taiwan cardholders, 2005 histories; 23 original predictors, six observed historical months, binary future-payment default label. 22.12% defaults. UCI source DOI 10.24432/C55S3H, CC BY 4.0. Raw data is not committed. Provenance and checksum are saved. IDs are excluded. Education and marriage ambiguous codes are recoded; genuine outliers and negative bill amounts are retained. Identical feature records remain grouped in splitting and CV.

## Evaluation and selection

Approximately 70/15/15 grouped-stratified partitions, seed 42. Hyperparameters optimize training-only grouped CV AUC. Validation AUC selects the overall winner and a separate tree for SHAP, if needed. Thresholds are set on validation. The final test is used solely for final descriptive reporting after choices are frozen. This is random within-population holdout validation, not out-of-time or external validation.

| Model | Validation AUC | Test AUC | Test AP | Test KS | Test Brier |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.7683 | 0.7663 | 0.5331 | 0.4277 | 0.1849 |
| decision_tree | 0.7298 | 0.7366 | 0.4815 | 0.3893 | 0.1377 |
| random_forest | 0.7779 | 0.7770 | 0.5479 | 0.4227 | 0.1351 |
| xgboost | 0.7822 | 0.7824 | 0.5534 | 0.4286 | 0.1341 |
| lightgbm | 0.7816 | 0.7841 | 0.5537 | 0.4329 | 0.1802 |

Winner: xgboost selected on validation; test strategy approves 56.56% with observed bad rate 9.98%.


Full metrics, confusion matrices, calibration bins and validation-versus-training comparisons live in `artifacts/metrics/results.json`; `comparison.csv` is the compact summary. Baseline/tuned comparisons are measured, not assumed improved. No confidence intervals or statistical dominance claims are made. Library versions and feature names are recorded for artifact compatibility.

## Calibration and decisions

Brier and calibration curves evaluate raw model probabilities. Class weights are considered inside CV and may impair calibration; no post-hoc calibrator has been fitted. The illustrative strategy targets ≤10% observed validation bad rate among simulated approvals, maximizes approval fraction, and fixes a configurable review boundary. This does not guarantee future defaults, realized profits or policy effectiveness. Test policy is frozen; dashboard sliders only use validation.

## Fairness and limitations

SEX, AGE, EDUCATION and MARRIAGE may be sensitive or act as proxies. Their inclusion reproduces an academic benchmark, not a justification for lending use. No subgroup fairness audit, protected-attribute compliance assessment or causal identification has been completed. A single historical population and existing-customer sampling constrain generalization. Credit policy, defaults and income conditions can shift. The dataset has no observed income, transaction-level histories, LGD, recovery, lending profit or decision counterfactuals. Repeated bill/payment amounts are not independent transaction flows.

SHAP and WOE explain associations or a model's computation. They cannot establish why a customer defaulted. PSI only diagnoses distribution changes; it does not certify acceptable predictive performance. The simulated monitoring shift is artificial and explicitly labeled. Only trusted locally generated Joblib artifacts should be loaded because deserialization executes code.

## Before real use

Obtain contemporary, authorized data; validate temporal/external generalization; audit subgroup discrimination/calibration and uncertainty; justify feature policy; evaluate calibrated candidates without test leakage; estimate exposure/loss/cost assumptions separately; implement security, auditability, monitoring, rollback and human governance. Those capabilities are not claimed here.
