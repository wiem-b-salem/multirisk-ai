# Methodology and Results

This document records the data, preprocessing, models, evaluation protocol,
results, and limitations of the MultiRisk AI MVP. All numbers are taken from the
source code, from CSV/TXT files under `results/`, from
`config/risk_thresholds.json`, or from the command output saved in
`results/cvd_diabetes_family_history_output.txt` and
`results/cvd_diabetes_thresholds_output.txt`.

## 1. Problem definition and scope

The system estimates the **current** risk of four diseases from a person's
family history, clinical measurements, genetic-risk information, and
demographic factors: Type 2 Diabetes, Cardiovascular Disease (CVD), Alzheimer's
disease, and liver disease. All four source datasets are cross-sectional: they
record current diagnosis status at a single time point. The models therefore
estimate current risk from current state; they do **not** predict future disease
onset.

The liver form reuses the first-input key `family_history_field`, but the field
is `GeneticRisk`, a genetic risk level from 0 to 2. It is not a family-history
question. The source does not say how that level is determined.

## 2. Datasets

### 2.1 Diabetes — Diabetes Health Dataset Analysis

| Field | Value |
|-------|-------|
| Where found | Kaggle |
| Link | https://www.kaggle.com/datasets/rabieelkharoua/diabetes-health-dataset-analysis |
| Access | Free (no institutional approval needed) |
| Academic use | Permitted (Kaggle CC BY 4.0 license) — *to be verified on the source page* |
| Rows | 1,879 records |
| Variables | 46 |
| Target | `Diagnosis` (0 = No: 1,127 / 60%; 1 = Yes: 752 / 40%) |
| Family history | `FamilyHistoryDiabetes` (0 = No: 1,431 / 76%; 1 = Yes: 448 / 24%) |
| Age range | 20–90 years |
| Missing values | 0 (100% complete) |
| Outcome type | Cross-sectional |

The diabetes dataset is **synthetic**. `docs/PHASE1_DATASET_RESEARCH.md` states
that its Kaggle description says it was generated for educational purposes and
that results do not reflect real patients.

### 2.2 CVD — Coronary Heart Disease (CHDdata)

| Field | Value |
|-------|-------|
| Where found | Kaggle |
| Link | https://www.kaggle.com/datasets/billbasener/coronary-heart-disease |
| Access | Free (no institutional approval needed) |
| Academic use | Permitted (license text not named in the research note) — *to be verified on the source page* |
| Rows | 462 records |
| Variables | 10 |
| Target | `chd` (0 = No: 302 / 65%; 1 = Yes: 160 / 35%) |
| Family history | `famhist` (Absent: 270 / 58%; Present: 192 / 42%) |
| Age range | 15–64 years |
| Missing values | 0 (100% complete) |
| Outcome type | Cross-sectional |

The CVD dataset is a retrospective case-control sample of **men aged 15–64 from
a high-risk region of South Africa** (from the `population_note` in
`src/export_artifacts.py`). Because it is case-control, model scores are **not
calibrated probabilities**.

### 2.3 Alzheimer's disease — Kaggle Alzheimer's Disease Dataset

| Field | Value |
|-------|-------|
| Where found | Kaggle |
| Link | Kaggle Alzheimer's Disease Dataset page (exact URL not recorded in the repository) |
| Access | Free (no institutional approval needed) |
| Academic use | *To be verified on the source page* |
| Licence and provenance | *To be verified on the source page* |
| Rows | 2,149 records |
| Variables | 35 (34 predictors and `Diagnosis`) |
| Target | `Diagnosis` (0 = No: 1,389 / 65%; 1 = Yes: 760 / 35%; positive rate 0.3537) |
| Family history | `FamilyHistoryAlzheimers` (0 = No: 1,607; 1 = Yes: 542; positive rate 0.2522) |
| Age range | 60–90 years |
| Missing values | 0 in every column |
| Outcome type | Cross-sectional |

`PatientID` is unique per row and `DoctorInCharge` is constant (`XXXConfid`), so
both are dropped. `Ethnicity` has values 0–3 and is one-hot encoded;
`EducationLevel` has values 0–3 and stays numeric. The meaning of the `Gender`
and `EducationLevel` codes is *to be verified on the source page*. `MMSE` ranges
from 0.0053 to 29.9914 and is nearly evenly spread from 0 to 30, so the dataset
appears synthetic; this is *to be verified on the source page*. Results do not
reflect real patients.

### 2.4 Liver disease — Liver disease dataset

| Field | Value |
|-------|-------|
| Where found | Kaggle source page (exact URL not recorded in the repository) |
| Academic use | *To be verified on the source page* |
| Licence and provenance | *To be verified on the source page* |
| Rows | 1,700 records |
| Variables | 11 |
| Target | `Diagnosis` (0 = No: 764; 1 = Yes: 936; positive rate 0.5506) |
| First form field | `GeneticRisk` (level 0: 978 rows; level 1: 557 rows; level 2: 165 rows) |
| Age range | 20–80 years |
| Missing values | 0 |
| Outcome type | Cross-sectional |

`GeneticRisk` is a genetic risk level, not a family-history question. The source
does not document the coding or units of `Gender`, `AlcoholConsumption`,
`GeneticRisk`, `PhysicalActivity`, or `LiverFunctionTest`; their meaning is *to
be verified on the source page*. The dataset appears synthetic; this is *to be
verified on the source page*. It is not validated on real patients.

### 2.5 Datasets considered and not used

From `docs/PHASE1_DATASET_RESEARCH.md`:

| Disease | Alternatives recorded | Status in the repository |
|---------|-----------------------|--------------------------|
| Diabetes | ARIC Study; Other | Not selected; no reason recorded beyond the selected dataset being feasible for the MVP |
| CVD | Framingham Heart Study; Other | Not selected; no reason recorded beyond the selected dataset being feasible for the MVP |
| Alzheimer's disease | PREVENT-AD; WRAP (Wisconsin Registry); Other | Not selected; no reason recorded |
| Liver disease | No candidate list recorded | Not applicable |

## 3. Preprocessing and feature sets

Common protocol (`src/cardiovascular/preprocess.py`,
`src/diabetes/preprocess.py`, `src/alzheimers/preprocess.py`, and
`src/liver/preprocess.py`): a stratified 80/20 train/test split with
`random_state=42`, and a `StandardScaler` fitted on the training set only and
then applied to both sets.

- **CVD preprocessing.** `famhist` is mapped from `Absent`/`Present` to `0`/`1`;
  the feature matrix is every column except the target `chd`.
- **Diabetes preprocessing.** `PatientID` and `DoctorInCharge` are always
  dropped. `Ethnicity` is one-hot encoded (`drop_first=True`) for the `full` and
  `risk_only` feature sets. The target is `Diagnosis`.
- **Alzheimer's preprocessing.** `PatientID` and `DoctorInCharge` are always
  dropped. `Ethnicity` is one-hot encoded (`drop_first=True`) for the `full` and
  `risk_only` feature sets. The target is `Diagnosis`.
- **Liver preprocessing.** The feature matrices are fixed column lists from
  `src/liver/preprocess.py`; `GeneticRisk` stays numeric (0/1/2), and the target
  is `Diagnosis`.

Three diabetes feature sets are defined in `src/diabetes/preprocess.py`:

| Feature set | Definition | Purpose |
|-------------|------------|---------|
| `full` | All features after dropping IDs and one-hot encoding `Ethnicity` | Full-information baseline |
| `risk_only` | `full` minus `FastingBloodSugar` and `HbA1c` | Measures performance without the two lab markers |
| `compact` | Fixed 12 columns: `Age`, `Gender`, `BMI`, `FamilyHistoryDiabetes`, `FastingBloodSugar`, `HbA1c`, `SystolicBP`, `DiastolicBP`, `Hypertension`, `PreviousPreDiabetes`, `FrequentUrination`, `ExcessiveThirst` | The served model's short, interpretable input form |

Three Alzheimer's feature sets are defined in `src/alzheimers/preprocess.py`:

| Feature set | Definition | Purpose |
|-------------|------------|---------|
| `full` | 34 features: all non-dropped columns except `Diagnosis`, with `Ethnicity` one-hot encoded | Maximum available signal, including clinician-scored assessments |
| `risk_only` | 24 features: `full` minus `MMSE`, `FunctionalAssessment`, `MemoryComplaints`, `BehavioralProblems`, `ADL`, `Confusion`, `Disorientation`, `PersonalityChanges`, `DifficultyCompletingTasks`, and `Forgetfulness` | Tests whether ordinary risk factors alone carry signal |
| `compact` | Fixed 12 columns: `Age`, `Gender`, `EducationLevel`, `BMI`, `FamilyHistoryAlzheimers`, `CardiovascularDisease`, `Diabetes`, `Depression`, `HeadInjury`, `Hypertension`, `MMSE`, `MemoryComplaints` | The served model's short form; `FunctionalAssessment` and `ADL` are excluded because a normal user cannot answer those clinician-scored scales |

Two liver feature sets are defined in `src/liver/preprocess.py`:

| Feature set | Definition | Purpose |
|-------------|------------|---------|
| `full` | 10 columns: `Age`, `Gender`, `BMI`, `AlcoholConsumption`, `Smoking`, `GeneticRisk`, `PhysicalActivity`, `Diabetes`, `Hypertension`, `LiverFunctionTest` | Maximum available signal; the set served by the API |
| `no_lab` | 9 columns: `full` minus `LiverFunctionTest` | Measures how much signal the laboratory value adds |

## 4. Models and evaluation protocol

Model definitions (`src/baselines.py`, `src/mlp.py`; the disease-specific
scripts use the same definitions):

- **logreg**: `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
- **rf**: `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)`.
- **mlp**: Keras `Sequential` — `Input(n_features)` → `Dense(64, relu)` →
  `Dropout(0.3)` → `Dense(32, relu)` → `Dropout(0.2)` → `Dense(1, sigmoid)`;
  optimizer `Adam(learning_rate=0.001)`; loss `binary_crossentropy`; metric
  `AUC`; balanced class weights; `EarlyStopping(monitor="val_loss", patience=15,
  restore_best_weights=True)`; `epochs=200`, `batch_size=32`.

Evaluation protocol:

- Stratified 80/20 train/test split, seed 42.
- Scaler fitted on the training set only.
- **5-fold stratified cross-validation on the training set** (`StratifiedKFold`,
  shuffle, seed 42, scoring `roc_auc`) → `cv_roc_auc_mean` and `cv_roc_auc_std`.
- **A single evaluation on the held-out test set** for all reported test metrics.
- The **final model per setup is chosen by highest CV AUC mean**.
- The MLP is additionally refit over seeds 0–4 to report
  `test_roc_auc_seed_mean` / `test_roc_auc_seed_std`.

## 5. Results

All metrics below are rounded from `results/summary.csv` (CVD, diabetes_full,
diabetes_risk_only), `results/diabetes_compact.csv` (diabetes_compact),
`results/alzheimers_summary.csv`, and `results/liver_summary.csv`.

| Setup | Model | CV AUC mean | CV AUC std | Test AUC | PR-AUC | Precision | Recall | Specificity | F1 |
|-------|-------|-------------|------------|----------|--------|-----------|--------|-------------|----|
| cvd | logreg | 0.7591 | 0.0650 | 0.8207 | 0.7141 | 0.5185 | 0.8750 | 0.5738 | 0.6512 |
| cvd | rf | 0.7135 | 0.0541 | 0.7218 | 0.5668 | 0.4444 | 0.6250 | 0.5902 | 0.5195 |
| cvd | mlp | 0.7432 | 0.0737 | 0.7956 | 0.6304 | 0.5200 | 0.8125 | 0.6066 | 0.6341 |
| diabetes_full | logreg | 0.8974 | 0.0141 | 0.9183 | 0.8922 | 0.7888 | 0.8467 | 0.8496 | 0.8167 |
| diabetes_full | rf | 0.9526 | 0.0110 | 0.9621 | 0.9419 | 0.9474 | 0.8400 | 0.9690 | 0.8905 |
| diabetes_full | mlp | 0.8840 | 0.0156 | 0.9171 | 0.8869 | 0.8025 | 0.8667 | 0.8584 | 0.8333 |
| diabetes_risk_only | logreg | 0.5731 | 0.0337 | 0.6102 | 0.5215 | 0.4590 | 0.5600 | 0.5619 | 0.5045 |
| diabetes_risk_only | rf | 0.5580 | 0.0256 | 0.5999 | 0.5054 | 0.5155 | 0.3333 | 0.7920 | 0.4049 |
| diabetes_risk_only | mlp | 0.5438 | 0.0289 | 0.5952 | 0.5011 | 0.4804 | 0.5733 | 0.5885 | 0.5228 |
| diabetes_compact | logreg | 0.9033 | 0.0132 | 0.9175 | 0.8837 | 0.7975 | 0.8400 | 0.8584 | 0.8182 |
| diabetes_compact | rf | 0.9574 | 0.0119 | 0.9547 | 0.9401 | 0.9301 | 0.8867 | 0.9558 | 0.9078 |
| alzheimers_full | logreg | 0.9033 | 0.0138 | 0.8841 | 0.8193 | 0.6895 | 0.8618 | 0.7878 | 0.7661 |
| alzheimers_full | rf | 0.9542 | 0.0107 | 0.9394 | 0.9259 | 0.9392 | 0.9145 | 0.9676 | 0.9267 |
| alzheimers_full | mlp | 0.8935 | 0.0128 | 0.8916 | 0.8382 | 0.7216 | 0.8355 | 0.8237 | 0.7744 |
| alzheimers_risk_only | logreg | 0.5411 | 0.0103 | 0.5282 | 0.4069 | 0.3705 | 0.5461 | 0.4928 | 0.4415 |
| alzheimers_risk_only | rf | 0.5243 | 0.0177 | 0.4861 | 0.3525 | 0.3488 | 0.1974 | 0.7986 | 0.2521 |
| alzheimers_risk_only | mlp | 0.5189 | 0.0194 | 0.4930 | 0.3499 | 0.3368 | 0.4276 | 0.5396 | 0.3768 |
| alzheimers_compact | logreg | 0.7254 | 0.0065 | 0.6922 | 0.5909 | 0.4645 | 0.5592 | 0.6475 | 0.5075 |
| alzheimers_compact | rf | 0.7243 | 0.0277 | 0.6955 | 0.5571 | 0.5610 | 0.4539 | 0.8058 | 0.5018 |
| alzheimers_compact | mlp | 0.7217 | 0.0224 | 0.7245 | 0.6005 | 0.5438 | 0.5724 | 0.7374 | 0.5577 |
| liver_full | logreg | 0.9037 | 0.0153 | 0.8957 | 0.9093 | 0.8621 | 0.8021 | 0.8431 | 0.8310 |
| liver_full | rf | 0.9370 | 0.0116 | 0.9599 | 0.9543 | 0.9379 | 0.8877 | 0.9281 | 0.9121 |
| liver_full | mlp | 0.9083 | 0.0154 | 0.9152 | 0.9276 | 0.8772 | 0.8021 | 0.8627 | 0.8380 |
| liver_no_lab | logreg | 0.8302 | 0.0077 | 0.8088 | 0.8374 | 0.7627 | 0.7219 | 0.7255 | 0.7418 |
| liver_no_lab | rf | 0.8501 | 0.0073 | 0.8361 | 0.8589 | 0.7672 | 0.7754 | 0.7124 | 0.7713 |
| liver_no_lab | mlp | 0.8372 | 0.0112 | 0.8155 | 0.8416 | 0.7552 | 0.7754 | 0.6928 | 0.7652 |

MLP seed stability (from the seed columns in the same CSVs):

- `cvd` test AUC seed mean 0.7874 ± 0.0181.
- `diabetes_full` 0.9115 ± 0.0063.
- `diabetes_risk_only` 0.5910 ± 0.0199.
- `alzheimers_full` 0.8855 ± 0.0040.
- `alzheimers_risk_only` 0.5082 ± 0.0164.
- `alzheimers_compact` 0.7148 ± 0.0100.
- `liver_full` 0.9052 ± 0.0046.
- `liver_no_lab` 0.8112 ± 0.0035.

**Final model per setup** (chosen by highest CV AUC mean):

| Setup | Final model | CV AUC mean | Served by API |
|-------|-------------|-------------|---------------|
| cvd | logreg | 0.7591 | Yes |
| diabetes_full | rf | 0.9526 | No |
| diabetes_risk_only | logreg | 0.5731 | No |
| diabetes_compact | rf | 0.9574 | Yes |
| alzheimers_full | rf | 0.9542 | No |
| alzheimers_risk_only | logreg | 0.5411 | No |
| alzheimers_compact | logreg | 0.7254 | Yes |
| liver_full | rf | 0.9370 | Yes |
| liver_no_lab | rf | 0.8501 | No |

The four served setups are recorded in `src/ml_api/main.py`: `cvd`,
`diabetes_compact`, `alzheimers_compact`, and `liver_full`.

**Figures** (`results/figures/`):

- CVD: [roc_cvd.png](../results/figures/roc_cvd.png),
  [confusion_cvd.png](../results/figures/confusion_cvd.png),
  [importance_cvd.png](../results/figures/importance_cvd.png),
  [calibration_cvd.png](../results/figures/calibration_cvd.png).
- diabetes_full: [roc_diabetes_full.png](../results/figures/roc_diabetes_full.png),
  [confusion_diabetes_full.png](../results/figures/confusion_diabetes_full.png),
  [importance_diabetes_full.png](../results/figures/importance_diabetes_full.png),
  [calibration_diabetes_full.png](../results/figures/calibration_diabetes_full.png).
- diabetes_risk_only: [roc_diabetes_risk_only.png](../results/figures/roc_diabetes_risk_only.png),
  [confusion_diabetes_risk_only.png](../results/figures/confusion_diabetes_risk_only.png),
  [importance_diabetes_risk_only.png](../results/figures/importance_diabetes_risk_only.png),
  [calibration_diabetes_risk_only.png](../results/figures/calibration_diabetes_risk_only.png).
- alzheimers_compact: [alzheimers_roc_alzheimers_compact.png](../results/figures/alzheimers_roc_alzheimers_compact.png),
  [alzheimers_confusion_alzheimers_compact.png](../results/figures/alzheimers_confusion_alzheimers_compact.png),
  [alzheimers_importance_alzheimers_compact.png](../results/figures/alzheimers_importance_alzheimers_compact.png),
  [alzheimers_calibration_alzheimers_compact.png](../results/figures/alzheimers_calibration_alzheimers_compact.png).
- liver_full: [liver_roc_liver_full.png](../results/figures/liver_roc_liver_full.png),
  [liver_confusion_liver_full.png](../results/figures/liver_confusion_liver_full.png),
  [liver_importance_liver_full.png](../results/figures/liver_importance_liver_full.png),
  [liver_calibration_liver_full.png](../results/figures/liver_calibration_liver_full.png).

There are no separate figures for `diabetes_compact`, `alzheimers_full`,
`alzheimers_risk_only`, or `liver_no_lab`; their numbers are in the CSV files.

## 6. Family-history analysis

From `results/cvd_diabetes_family_history_output.txt`:

CVD (`famhist`):

- Positive rate, family history absent: 0.237.
- Positive rate, family history present: 0.500.
- Odds ratio: 3.219.
- Chi-square p-value: 8.6527e-09.

Diabetes (`FamilyHistoryDiabetes`):

- Positive rate, family history absent: 0.387.
- Positive rate, family history present: 0.442.
- Odds ratio: 1.254.
- Chi-square p-value: 0.0442618.

Standardized logistic-regression coefficients for the family-history feature:

- `cvd`: 0.4028.
- `diabetes_full`: 0.0669.
- `diabetes_risk_only`: 0.0855.

From `results/alzheimers_family_history.txt`, using
`FamilyHistoryAlzheimers`:

- Positive rate, family history absent: 0.363.
- Positive rate, family history present: 0.327.
- Odds ratio: 0.852.
- Chi-square p-value: 0.140698.
- Standardized coefficients: `full` -0.0657, `risk_only` -0.0649, `compact` -0.0634.

Family history is a strong CVD signal (odds ratio 3.219, p-value 8.6527e-09), a
weak diabetes signal (odds ratio 1.254, p-value 0.0442618), and shows no positive
effect for Alzheimer's disease in this data (odds ratio 0.852, p-value 0.140698,
with negative standardized coefficients in all three feature sets). Liver
disease has no family-history variable; see the genetic-risk analysis in
Section 8.

## 7. Threshold calibration

Method: out-of-fold predictions are produced on the training set with
`cross_val_predict` (5-fold, seed 42). For CVD, Alzheimer's disease, and liver
disease, `low_max` is the largest threshold with recall ≥ 0.90 and `high_min` is
the smallest threshold with specificity ≥ 0.90. For diabetes, `low_max` would be
≥ `high_min`, so a single Youden's J threshold is used, giving two levels
(low/high).

Chosen thresholds (`config/risk_thresholds.json`):

- `cvd`: `low_max` = 0.27703360102750624, `high_min` = 0.711124446390867.
- `diabetes`: `low_max` = 0.44, `high_min` = 0.44.
- `alzheimers`: `low_max` = 0.2985038875290284, `high_min` = 0.6448177527418835.
- `liver`: `low_max` = 0.47333333333333333, `high_min` = 0.54.

CVD, Alzheimer's disease, and liver disease have three bands because `low_max <
high_min`. Diabetes has two bands because `low_max == high_min`.

**CVD out-of-fold bands** (`results/cvd_diabetes_thresholds_output.txt`):

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 105 | 0.284553 | 0.114286 |
| medium | 188 | 0.509485 | 0.340426 |
| high | 76 | 0.205962 | 0.684211 |

**Diabetes out-of-fold bands** (`results/cvd_diabetes_thresholds_output.txt`):

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 921 | 0.612774 | 0.071661 |
| high | 582 | 0.387226 | 0.920962 |

**Alzheimer's disease out-of-fold bands** (`docs/ALZHEIMERS_RESULTS.md`):

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 412 | 0.239674 | 0.145631 |
| medium | 949 | 0.552065 | 0.317176 |
| high | 358 | 0.208261 | 0.689944 |

**Liver disease out-of-fold bands** (`results/liver_thresholds_output.txt`):

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 601 | 0.441912 | 0.123128 |
| medium | 58 | 0.042647 | 0.568966 |
| high | 701 | 0.515441 | 0.915835 |

**CVD one-time test check (frozen thresholds):**

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 25 | 0.268817 | 0.040000 |
| medium | 38 | 0.408602 | 0.315789 |
| high | 30 | 0.322581 | 0.633333 |

**Diabetes one-time test check (frozen thresholds):**

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 226 | 0.601064 | 0.053097 |
| high | 150 | 0.398936 | 0.920000 |

**Alzheimer's disease one-time test check** (`docs/ALZHEIMERS_RESULTS.md`):

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 101 | 0.234884 | 0.148515 |
| medium | 233 | 0.541860 | 0.334764 |
| high | 96 | 0.223256 | 0.614583 |

**Liver disease one-time test check** (`results/liver_thresholds_output.txt`):

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 163 | 0.479412 | 0.128834 |
| medium | 15 | 0.044118 | 0.733333 |
| high | 162 | 0.476471 | 0.956790 |

`src/calibrate_thresholds` writes only `cvd`, `diabetes`, and `note` to
`config/risk_thresholds.json`; rerunning it therefore drops the `alzheimers` and
`liver` keys. Restore those keys (or run `src.alzheimers.calibrate_thresholds`
and `src.liver.calibrate_thresholds`) after running it. The disease-specific
calibration scripts preserve the other keys and assert that they are unchanged.

## 8. Liver genetic-risk analysis

From `results/liver_genetic_risk.txt`. Positive rate by `GeneticRisk` level:

| Level | Rows | Positive | Rate |
|-------|------|----------|------|
| 0 | 978 | 519 | 0.5307 |
| 1 | 557 | 279 | 0.5009 |
| 2 | 165 | 138 | 0.8364 |

The chi-square test across levels 0/1/2 gives chi2=61.5836, dof=2,
p-value=4.23935e-14. Level 1 versus 0: odds ratio 0.8876, p-value 0.284769.
Level 2 versus 0: odds ratio 4.5202, p-value 3.81593e-13. The standardized
logistic-regression coefficient of `GeneticRisk` is 0.4670 in `full` and 0.3603
in `no_lab`.

Added value in training-set CV ROC-AUC (same protocol), with versus without
`GeneticRisk`:

| Setup | With | Without |
|-------|------|---------|
| full/logreg | 0.9037 ± 0.0153 | 0.8946 ± 0.0170 |
| full/rf | 0.9370 ± 0.0116 | 0.9197 ± 0.0124 |
| no_lab/logreg | 0.8302 ± 0.0077 | 0.8216 ± 0.0083 |
| no_lab/rf | 0.8501 ± 0.0073 | 0.8388 ± 0.0084 |

These numbers support a higher positive rate at level 2 and a small CV gain
from keeping the variable. They do not establish what the level means, because
the source does not document its coding.

## 9. Limitations

- **Synthetic data.** The diabetes dataset is synthetic; the Alzheimer's and
  liver datasets appear synthetic, which is *to be verified on the source page*.
  Results do not reflect real patients.
- **Diabetes performance is driven almost entirely by the lab markers.**
  `HbA1c` and `FastingBloodSugar` dominate the compact model's importances.
- **Near-chance performance without them.** Removing both markers
  (`diabetes_risk_only`) drops test AUC to 0.6102 for logreg, 0.5999 for rf, and
  0.5952 for the MLP.
- **Family-history strength varies.** Odds ratio 3.219 for CVD, 1.254 for
  diabetes, and 0.852 for Alzheimer's disease. The Alzheimer's result is not a
  positive effect.
- **CVD sample is small, male only, aged 15–64, and case-control.** Scores are
  not probabilities, and the model is not validated for women or for ages
  outside 15–64. The dataset provenance is *to be verified on the source page*.
- **Alzheimer's performance depends heavily on cognitive and functional
  assessments.** The full set reaches CV AUC 0.9542 for random forest, while the
  same models without those assessments are near chance (0.5411 for logistic
  regression and 0.5243 for random forest).
- **The deployable Alzheimer's set is weaker by design.** `alzheimers_compact`
  reaches CV AUC 0.7254 and excludes `FunctionalAssessment` and `ADL`, which are
  clinician-scored scales a normal user cannot answer. Only people aged 60–90
  are represented.
- **Liver units and genetic-risk meaning are unclear.** The source does not
  document the coding or units of several fields, and `GeneticRisk` is a
  genetic-risk level, not a family-history question.
- **The MLP never beat the best classical model on the selection metric.** By CV
  AUC, the best setup result always came from logreg or random forest. A higher
  test AUC for one MLP, such as 0.7245 for `alzheimers_compact`, was not used for
  selection.
- **The diabetes model is two-level only.** `low_max == high_min == 0.44`, so
  only low/high bands exist.
- **Random-forest models return no top factors.** The served diabetes and liver
  models are random forests, so the API returns `top_factors: null` for them.
- **Unclear units in the source.** `tobacco`, `adiposity`, and `alcohol` have
  unclear units in the CVD dataset (noted in the field help text).
- **No longitudinal outcome.** All four datasets are cross-sectional, so the
  models estimate current risk, not future risk.

## 10. Reproducibility

Run from the repository root with the project virtual environment. The CSV
files may need to be downloaded from their source pages and placed under
`data/`; `data/liver/Liver_disease_data.csv` is noted in
`docs/LIVER_RESULTS.md`.

```powershell
.\venv\Scripts\python.exe -m src.baselines
.\venv\Scripts\python.exe -m src.mlp
.\venv\Scripts\python.exe -m src.report
.\venv\Scripts\python.exe -m src.diabetes_compact
.\venv\Scripts\python.exe -m src.family_history_check
.\venv\Scripts\python.exe -m src.calibrate_thresholds
.\venv\Scripts\python.exe -m src.export_artifacts
.\venv\Scripts\python.exe -m src.alzheimers.eda
.\venv\Scripts\python.exe -m src.alzheimers.baselines
.\venv\Scripts\python.exe -m src.alzheimers.mlp
.\venv\Scripts\python.exe -m src.alzheimers.family_history_check
.\venv\Scripts\python.exe -m src.alzheimers.report
.\venv\Scripts\python.exe -m src.alzheimers.calibrate_thresholds
.\venv\Scripts\python.exe -m src.alzheimers.export_artifacts
.\venv\Scripts\python.exe -m src.liver.eda
.\venv\Scripts\python.exe -m src.liver.baselines
.\venv\Scripts\python.exe -m src.liver.mlp
.\venv\Scripts\python.exe -m src.liver.genetic_risk_check
.\venv\Scripts\python.exe -m src.liver.importance_check
.\venv\Scripts\python.exe -m src.liver.report
.\venv\Scripts\python.exe -m src.liver.calibrate_thresholds
.\venv\Scripts\python.exe -m src.liver.export_artifacts
.\venv\Scripts\python.exe -m pytest -q
```

- The baseline, MLP, and report scripts write the metric CSVs, final joblib
  models, and figures described above.
- `src.family_history_check` prints the CVD/diabetes family-history numbers.
- `src.calibrate_thresholds` rewrites `config/risk_thresholds.json` with only the
  `cvd`, `diabetes`, and `note` keys; restore the other keys if needed.
- `src.alzheimers.family_history_check` writes
  `results/alzheimers_family_history.txt`.
- `src.liver.genetic_risk_check` writes `results/liver_genetic_risk.txt`.
- `src.liver.importance_check` writes `results/liver_importance_output.txt`.
- The report scripts need their `mlp_*.keras` files, so run the matching MLP
  script first. The keras files are not tracked in the repository. The liver
  importance numbers can be reproduced without that file by running
  `src.liver.importance_check`.
- Each export script rewrites its own schema JSON files under `models/`.

## 13. Deployed ML versus DL comparison

The deployed comparison is produced by `src/deep_serving.py`, not by the research
MLP scripts. It trains one MLP per served feature set, evaluates it on the same
held-out test split as the preserved ML model, and derives separate DL thresholds
from five-fold out-of-fold predictions on the training split only.

| Disease | ML model | ML ROC-AUC | ML accuracy | DL ROC-AUC | DL accuracy | DL training time |
|---------|----------|------------|-------------|------------|-------------|------------------|
| CVD | Logistic Regression | 0.8207 | 0.6774 | 0.7956 | 0.6774 | 5.48 s |
| Diabetes | Random Forest | 0.9547 | 0.9282 | 0.9456 | 0.9016 | 10.74 s |
| Alzheimer's | Logistic Regression | 0.6922 | 0.6163 | 0.7240 | 0.6698 | 7.51 s |
| Liver | Random Forest | 0.9599 | 0.9059 | 0.9152 | 0.8294 | 11.29 s |

Full numbers are in `results/comparison_metrics.csv`. The API reads
`config/comparison_metrics.json` and `config/dl_risk_thresholds.json`. The served
MLP files are explicitly unignored in `.gitignore`, while research MLP variants
remain untracked. See `docs/COMPARISON_REPORT.md` for the full final report.
