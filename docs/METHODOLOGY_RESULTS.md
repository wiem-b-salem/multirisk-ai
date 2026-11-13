# Methodology and Results

This document records the data, preprocessing, models, evaluation protocol,
results, and limitations of the MultiRisk AI MVP. All numbers are taken from the
source code, from `results/baselines.csv`, `results/mlp.csv`,
`results/summary.csv`, `results/diabetes_compact.csv`,
`config/risk_thresholds.json`, or from the output of
`python -m src.family_history_check` and `python -m src.calibrate_thresholds`.

## 1. Problem definition and scope

The system estimates the **current** risk of Type 2 Diabetes and Cardiovascular
Disease (CVD) from a person's family history, clinical measurements, and
demographic factors. Both source datasets are cross-sectional: they record
current diagnosis status at a single time point. The models therefore estimate
current risk from current state; they do **not** predict future disease onset.
This scope is stated in `README.md` and in `docs/PHASE1_DATASET_RESEARCH.md`
("Cross-sectional data (current diagnosis, not future risk prediction)").

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

The licensing text above is copied from `docs/PHASE1_DATASET_RESEARCH.md` and is
marked "to be verified on the source page". The diabetes dataset is synthetic;
the CVD dataset is a retrospective case-control sample of men aged 15–64 from a
high-risk region of South Africa.

## 3. Preprocessing and diabetes feature sets

Common protocol (`src/cardiovascular/preprocess.py`,
`src/diabetes/preprocess.py`): a stratified 80/20 train/test split with
`random_state=42`, and a `StandardScaler` fitted on the training set only and
then applied to both sets.

- **CVD preprocessing.** `famhist` is mapped from `Absent`/`Present` to `0`/`1`;
  the feature matrix is every column except the target `chd`.
- **Diabetes preprocessing.** `PatientID` and `DoctorInCharge` are always
  dropped. `Ethnicity` is one-hot encoded (`drop_first=True`) for the `full` and
  `risk_only` feature sets. The target is `Diagnosis`.

Three diabetes feature sets are defined in `src/diabetes/preprocess.py`:

| Feature set | Definition | Purpose |
|-------------|------------|---------|
| `full` | All features after dropping IDs and one-hot encoding `Ethnicity` | Full-information baseline |
| `risk_only` | `full` minus `FastingBloodSugar` and `HbA1c` | Measures performance without the two lab markers |
| `compact` | Fixed 12 columns: `Age`, `Gender`, `BMI`, `FamilyHistoryDiabetes`, `FastingBloodSugar`, `HbA1c`, `SystolicBP`, `DiastolicBP`, `Hypertension`, `PreviousPreDiabetes`, `FrequentUrination`, `ExcessiveThirst` | The served model's short, interpretable input form |

The `compact` set keeps the clinically meaningful inputs needed for a short form
while remaining close to the `full` set in performance (Section 5).

## 4. Models and evaluation protocol

Model definitions (`src/baselines.py`, `src/mlp.py`):

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
- The **final model per setup is chosen by highest CV AUC mean** (`src/report.py`,
  `src/diabetes_compact.py`).
- The MLP is additionally refit over seeds 0–4 to report
  `test_roc_auc_seed_mean` / `test_roc_auc_seed_std`.

## 5. Results

All metrics below are from `results/summary.csv` (CVD, diabetes_full,
diabetes_risk_only) and `results/diabetes_compact.csv` (diabetes_compact).

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

MLP seed stability (from `results/summary.csv`):
`cvd` test AUC seed mean 0.7874 ± 0.0181; `diabetes_full` 0.9115 ± 0.0063;
`diabetes_risk_only` 0.5910 ± 0.0199.

**Final model per setup** (chosen by highest CV AUC mean):

| Setup | Final model | CV AUC mean |
|-------|-------------|-------------|
| cvd | logreg | 0.7591 |
| diabetes_full | rf | 0.9526 |
| diabetes_risk_only | logreg | 0.5731 |
| diabetes_compact | rf | 0.9574 |

The served diabetes model is the `diabetes_compact` Random Forest; the served
CVD model is the `cvd` logistic regression (both recorded in the schema JSON
files under `models/`).

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

There are no separate figures for `diabetes_compact`; its numbers are in
`results/diabetes_compact.csv`.

## 6. Family-history analysis

From `python -m src.family_history_check`:

CVD (`famhist`):

- Positive rate, family history absent: 0.237
- Positive rate, family history present: 0.500
- Odds ratio: 3.219
- Chi-square p-value: 8.6527e-09

Diabetes (`FamilyHistoryDiabetes`):

- Positive rate, family history absent: 0.387
- Positive rate, family history present: 0.442
- Odds ratio: 1.254
- Chi-square p-value: 0.0442618

Standardized logistic-regression coefficients for the family-history feature:

- `cvd`: 0.4028
- `diabetes_full`: 0.0669
- `diabetes_risk_only`: 0.0855

Family history is a strong CVD signal (odds ratio 3.219, highly significant) but
only a weak diabetes signal (odds ratio 1.254, p 0.0442618).

## 7. Threshold calibration

Method (`src/calibrate_thresholds.py`): out-of-fold predictions are produced on
the training set with `cross_val_predict` (5-fold, seed 42). For CVD, `low_max`
is the largest threshold with recall ≥ 0.90 and `high_min` is the smallest
threshold with specificity ≥ 0.90. For diabetes, `low_max` would be ≥ `high_min`,
so a single Youden's J threshold is used, giving two levels (low/high).

Chosen thresholds (`config/risk_thresholds.json`):

- `cvd`: `low_max` = 0.27703360102750624, `high_min` = 0.711124446390867
- `diabetes`: `low_max` = 0.44, `high_min` = 0.44

**CVD out-of-fold bands:**

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 105 | 0.284553 | 0.114286 |
| medium | 188 | 0.509485 | 0.340426 |
| high | 76 | 0.205962 | 0.684211 |

**Diabetes out-of-fold bands:**

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 921 | 0.612774 | 0.071661 |
| high | 582 | 0.387226 | 0.920962 |

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

## 8. Limitations

- **Synthetic diabetes data.** The diabetes dataset is synthetic; results do not
  reflect real patients.
- **Diabetes performance is driven almost entirely by the lab markers.**
  `HbA1c` and `FastingBloodSugar` dominate the compact model's importances.
- **Near-chance performance without them.** Removing both markers
  (`diabetes_risk_only`) drops test AUC to about 0.60 (logreg 0.6102, rf 0.5999,
  mlp 0.5952).
- **Family history is weak for diabetes and strong for CVD.** Odds ratio 1.254
  (diabetes) versus 3.219 (CVD).
- **CVD sample is small, male only, aged 15–64, and case-control.** The scores
  are not probabilities, and the model is not validated for women or for ages
  outside 15–64.
- **The MLP never beat the baselines.** On every setup the neural network's CV
  and test AUC were below the best classical model (for example CVD test AUC:
  mlp 0.7956 versus logreg 0.8207).
- **The diabetes model is two-level only.** `low_max == high_min == 0.44`, so
  only low/high bands exist.
- **Random Forest cannot extrapolate beyond the training range.** Values outside
  `train_min`/`train_max` are flagged with `out_of_training_range` and a
  `range_warning`.
- **Unclear units in the source.** `tobacco`, `adiposity`, and `alcohol` have
  unclear units in the CVD dataset (noted in the field help text).
- **No longitudinal outcome.** Both datasets are cross-sectional, so the models
  estimate current risk, not future risk.

## 9. Reproducibility

Run from the repository root with the project virtual environment, in this order:

```powershell
.\venv\Scripts\python.exe -m src.baselines
.\venv\Scripts\python.exe -m src.mlp
.\venv\Scripts\python.exe -m src.report
.\venv\Scripts\python.exe -m src.diabetes_compact
.\venv\Scripts\python.exe -m src.family_history_check
.\venv\Scripts\python.exe -m src.calibrate_thresholds
.\venv\Scripts\python.exe -m src.export_artifacts
```

- `src.baselines` writes `results/baselines.csv`.
- `src.mlp` writes `results/mlp.csv` and the `mlp_*.keras` files.
- `src.report` merges them into `results/summary.csv`, writes the final
  `final_*.joblib` models, and regenerates `results/figures/`.
- `src.diabetes_compact` writes `results/diabetes_compact.csv` and
  `models/diabetes/final_diabetes_compact.joblib`.
- `src.family_history_check` prints the family-history numbers.
- `src.calibrate_thresholds` rewrites `config/risk_thresholds.json` (it is
  reproducible: rerunning leaves the file unchanged).
- `src.export_artifacts` rewrites the schema JSON files under `models/`.
