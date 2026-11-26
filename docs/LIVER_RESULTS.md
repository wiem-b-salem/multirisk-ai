# Liver disease: methodology and results

## Dataset

`data/liver/Liver_disease_data.csv`: 1,700 rows and 11 columns, with no missing values. The target is `Diagnosis` (764 negative, 936 positive; positive rate 0.5506). Ages run from 20 to 80. `GeneticRisk` takes values 0 (978 rows), 1 (557 rows) and 2 (165 rows). It is a **Genetic risk level**, NOT a family-history question; the source does not say how the level is determined. Licence and provenance are to be verified on the source page.

## Feature sets

| Set | Features | Why |
|-----|----------|-----|
| `full` | 10: `Age`, `Gender`, `BMI`, `AlcoholConsumption`, `Smoking`, `GeneticRisk`, `PhysicalActivity`, `Diabetes`, `Hypertension`, `LiverFunctionTest` | Maximum available signal; the set served by the API |
| `no_lab` | 9: `full` minus `LiverFunctionTest` | Tests how much signal the laboratory value adds |

`GeneticRisk` stays numeric (0/1/2); no other encoding is needed.

## Protocol

Every set uses a stratified 80/20 split with `random_state=42` (1,360 training rows, 340 test rows) and a `StandardScaler` fitted on the training rows only. Logistic regression (`max_iter=1000`, `class_weight="balanced"`, `random_state=42`) and random forest (`n_estimators=300`, `class_weight="balanced"`, `random_state=42`) use the same five-fold stratified CV on the training set (shuffle, seed 42), with `cv_roc_auc_mean` as the only selection criterion. The MLP uses the same architecture and settings as the Alzheimer's model (64 and 32 hidden units, dropout 0.3 and 0.2, Adam at 0.001, binary crossentropy, balanced class weights, 200 epochs, batch size 32, early stopping with patience 15); its seed mean and standard deviation use seeds 0-4, and the seed-42 model is saved. The test set was used once for reporting, never for model, feature or threshold choices.

## Results

Rounded from `results/liver_summary.csv`. Counts are tn/fp/fn/tp at a 0.5 threshold.

| Setup | Model | CV ROC-AUC mean ± std | Test ROC-AUC | Test PR-AUC | Precision | Recall | Specificity | F1 | Counts | Seed mean ± std |
|-------|-------|-----------------------|--------------|-------------|-----------|--------|-------------|----|--------|-----------------|
| full | logreg | 0.9037 ± 0.0153 | 0.8957 | 0.9093 | 0.8621 | 0.8021 | 0.8431 | 0.8310 | 129/24/37/150 | - |
| full | rf | 0.9370 ± 0.0116 | 0.9599 | 0.9543 | 0.9379 | 0.8877 | 0.9281 | 0.9121 | 142/11/21/166 | - |
| full | mlp | 0.9083 ± 0.0154 | 0.9152 | 0.9276 | 0.8772 | 0.8021 | 0.8627 | 0.8380 | 132/21/37/150 | 0.9052 ± 0.0046 |
| no_lab | logreg | 0.8302 ± 0.0077 | 0.8088 | 0.8374 | 0.7627 | 0.7219 | 0.7255 | 0.7418 | 111/42/52/135 | - |
| no_lab | rf | 0.8501 ± 0.0073 | 0.8361 | 0.8589 | 0.7672 | 0.7754 | 0.7124 | 0.7713 | 109/44/42/145 | - |
| no_lab | mlp | 0.8372 ± 0.0112 | 0.8155 | 0.8416 | 0.7552 | 0.7754 | 0.6928 | 0.7652 | 106/47/42/145 | 0.8112 ± 0.0035 |

## Final model

The full-set random forest was chosen by `cv_roc_auc_mean` (0.9370, against 0.9037 for logistic regression); the MLP (0.9083) is reported but not served, as for the other diseases. It is stored in `models/liver/final_liver_full.joblib` with `models/liver/scaler_liver_full.joblib`. On the test set: ROC-AUC 0.9599, PR-AUC 0.9543, precision 0.9379, recall 0.8877, specificity 0.9281, F1 0.9121, counts 142/11/21/166. Top five permutation importances on the test set (`n_repeats=20`, `random_state=42`, ROC-AUC drop):

| Feature | Mean ± std |
|---------|------------|
| LiverFunctionTest | 0.1819 ± 0.0174 |
| AlcoholConsumption | 0.1809 ± 0.0166 |
| Age | 0.0448 ± 0.0075 |
| Smoking | 0.0380 ± 0.0049 |
| GeneticRisk | 0.0334 ± 0.0044 |

Figures with the `liver_` prefix are in `results/figures/`: `liver_roc_liver_full.png`, `liver_calibration_liver_full.png`, `liver_confusion_liver_full.png` and `liver_importance_liver_full.png`.

## Genetic risk analysis

From `results/liver_genetic_risk.txt`. Positive rate by Genetic risk level: level 0, n=978, 519 positive (0.5307); level 1, n=557, 279 positive (0.5009); level 2, n=165, 138 positive (0.8364). The chi-square test across the three levels gives chi2=61.5836, dof=2, p=4.23935e-14. Level 1 versus 0: odds ratio 0.8876, p=0.284769. Level 2 versus 0: odds ratio 4.5202, p=3.81593e-13. The standardized logistic-regression coefficient of `GeneticRisk` is 0.4670 in `full` and 0.3603 in `no_lab`.

Added value in training-set CV ROC-AUC (same protocol), with versus without `GeneticRisk`: full/logreg 0.9037 ± 0.0153 versus 0.8946 ± 0.0170; full/rf 0.9370 ± 0.0116 versus 0.9197 ± 0.0124; no_lab/logreg 0.8302 ± 0.0077 versus 0.8216 ± 0.0083; no_lab/rf 0.8501 ± 0.0073 versus 0.8388 ± 0.0084. The numbers support a higher positive rate at level 2 and a small CV gain from keeping the variable.

## Risk thresholds

Thresholds come from out-of-fold probabilities on the training rows only (five-fold stratified, shuffle, seed 42). `config/risk_thresholds.json` records `low_max=0.47333333333333333` and `high_min=0.54`: `low_max` is the highest threshold with recall >= 0.90, `high_min` the lowest with specificity >= 0.90. Since `low_max < high_min`, there are three bands.

Training out-of-fold bands:

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 601 | 0.441912 | 0.123128 |
| medium | 58 | 0.042647 | 0.568966 |
| high | 701 | 0.515441 | 0.915835 |

## Limitations

- The dataset is synthetic and not validated on real patients; there is no external validation.
- The source does not document the coding or units of `Gender`, `AlcoholConsumption`, `GeneticRisk`, `PhysicalActivity` or `LiverFunctionTest`; the meaning of `GeneticRisk` in particular needs verification.
- `GeneticRisk` is a Genetic risk level, not a family-history question.
- This is an academic estimate, not a diagnostic tool or a calibrated individual probability.
