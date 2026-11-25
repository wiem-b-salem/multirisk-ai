# Alzheimer's disease: methodology and results

## Dataset

Kaggle Alzheimer's Disease Dataset: 2,149 rows and 35 columns, with no missing values. The target is `Diagnosis` (1,389 negative, 760 positive; positive rate 0.3537). `FamilyHistoryAlzheimers` is coded 0/1 (1,607 negative, 542 positive; positive rate 0.2522). Ages run from 60 to 90. `PatientID` is an identifier and `DoctorInCharge` is constant (`XXXConfid`), so both are dropped. `Ethnicity` has values 0-3 and is one-hot encoded with `drop_first=True`; `EducationLevel` has values 0-3 and stays numeric. `MMSE` ranges from 0.0053 to 29.9914 and is nearly evenly spread from 0 to 30, so the dataset appears synthetic. It is cross-sectional. Licence and provenance are to be verified on the source page.

## Feature sets

| Set | Features | Why |
|-----|----------|-----|
| `full` | 34: all non-dropped columns except `Diagnosis`, with `Ethnicity` one-hot encoded | Maximum available signal, including clinician-scored assessments |
| `risk_only` | 24: `full` minus `MMSE`, `FunctionalAssessment`, `MemoryComplaints`, `BehavioralProblems`, `ADL`, `Confusion`, `Disorientation`, `PersonalityChanges`, `DifficultyCompletingTasks`, `Forgetfulness` | Tests whether ordinary risk factors alone carry signal |
| `compact` | 12: `Age`, `Gender`, `EducationLevel`, `BMI`, `FamilyHistoryAlzheimers`, `CardiovascularDisease`, `Diabetes`, `Depression`, `HeadInjury`, `Hypertension`, `MMSE`, `MemoryComplaints` | Fixed user-input set served by the API. `FunctionalAssessment` and `ADL` are excluded because they are clinician-scored scales a normal user cannot answer |

Every set uses a stratified 80/20 split with `random_state=42` (1,719 training rows, 430 test rows) and a `StandardScaler` fitted on the training rows only. The test set was used once for reporting, never for model, feature or threshold choices.

## Models and protocol

Logistic regression (`max_iter=1000`, `class_weight="balanced"`, `random_state=42`), random forest (`n_estimators=300`, `class_weight="balanced"`, `random_state=42`) and an MLP with the same architecture and settings as `src/mlp.py` (64 and 32 hidden units, dropout 0.3 and 0.2, Adam at 0.001, binary crossentropy, balanced class weights, 200 epochs, batch size 32, early stopping with patience 15). Five-fold stratified CV on the training set (shuffle, seed 42) supplies `cv_roc_auc_mean`; the model is chosen by that value only. The MLP seed mean and standard deviation use seeds 0-4; the seed-42 model is saved.

## Results

Rounded from `results/alzheimers_summary.csv`. Counts are tn/fp/fn/tp at a 0.5 threshold.

| Setup | Model | CV ROC-AUC mean ± std | Test ROC-AUC | Test PR-AUC | Precision | Recall | Specificity | F1 | Counts | Seed mean ± std |
|-------|-------|-----------------------|--------------|-------------|-----------|--------|-------------|----|--------|-----------------|
| full | logreg | 0.9033 ± 0.0138 | 0.8841 | 0.8193 | 0.6895 | 0.8618 | 0.7878 | 0.7661 | 219/59/21/131 | - |
| full | rf | 0.9542 ± 0.0107 | 0.9394 | 0.9259 | 0.9392 | 0.9145 | 0.9676 | 0.9267 | 269/9/13/139 | - |
| full | mlp | 0.8935 ± 0.0128 | 0.8916 | 0.8382 | 0.7216 | 0.8355 | 0.8237 | 0.7744 | 229/49/25/127 | 0.8855 ± 0.0040 |
| risk_only | logreg | 0.5411 ± 0.0103 | 0.5282 | 0.4069 | 0.3705 | 0.5461 | 0.4928 | 0.4415 | 137/141/69/83 | - |
| risk_only | rf | 0.5243 ± 0.0177 | 0.4861 | 0.3525 | 0.3488 | 0.1974 | 0.7986 | 0.2521 | 222/56/122/30 | - |
| risk_only | mlp | 0.5189 ± 0.0194 | 0.4930 | 0.3499 | 0.3368 | 0.4276 | 0.5396 | 0.3768 | 150/128/87/65 | 0.5082 ± 0.0164 |
| compact | logreg | 0.7254 ± 0.0065 | 0.6922 | 0.5909 | 0.4645 | 0.5592 | 0.6475 | 0.5075 | 180/98/67/85 | - |
| compact | rf | 0.7243 ± 0.0277 | 0.6955 | 0.5571 | 0.5610 | 0.4539 | 0.8058 | 0.5018 | 224/54/83/69 | - |
| compact | mlp | 0.7217 ± 0.0224 | 0.7245 | 0.6005 | 0.5438 | 0.5724 | 0.7374 | 0.5577 | 205/73/65/87 | 0.7148 ± 0.0100 |

## Final model

The compact logistic regression was chosen by `cv_roc_auc_mean` (0.7254, against 0.7243 for random forest and 0.7217 for the MLP). It is stored in `models/alzheimers/final_alzheimers_compact.joblib` with `models/alzheimers/scaler_alzheimers_compact.joblib`. Top five permutation importances on the test set (`n_repeats=20`, `random_state=42`, ROC-AUC drop):

| Feature | Mean ± std |
|---------|------------|
| MMSE | 0.0824 ± 0.0152 |
| MemoryComplaints | 0.0803 ± 0.0159 |
| Hypertension | 0.0025 ± 0.0034 |
| EducationLevel | 0.0024 ± 0.0032 |
| CardiovascularDisease | 0.0017 ± 0.0030 |

## Family history

From `results/alzheimers_family_history.txt`:

```
=== Alzheimer's (FamilyHistoryAlzheimers) ===
Positive rate family history absent:  0.363
Positive rate family history present: 0.327
Odds ratio: 0.852
Chi-square p-value: 0.140698
=== Logistic regression coefficients (standardized) ===
full: -0.0657
risk_only: -0.0649
compact: -0.0634
```

## Risk thresholds

Thresholds come from out-of-fold probabilities on the training rows only (5-fold stratified, shuffle, seed 42): `low_max=0.2985038875290284` and `high_min=0.6448177527418835`, giving three levels because `low_max < high_min`.

Training out-of-fold bands:

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 412 | 0.239674 | 0.145631 |
| medium | 949 | 0.552065 | 0.317176 |
| high | 358 | 0.208261 | 0.689944 |

One-time test-set check with frozen thresholds:

| Band | Rows | Share | Positive rate |
|------|------|-------|---------------|
| low | 101 | 0.234884 | 0.148515 |
| medium | 233 | 0.541860 | 0.334764 |
| high | 96 | 0.223256 | 0.614583 |

## Limitations

- The dataset appears synthetic and is not validated on real patients.
- Performance depends heavily on cognitive and functional assessments: the full set reaches CV ROC-AUC 0.9542 (random forest), while the same models without those assessments are near chance (0.5411 for logistic regression and 0.5243 for random forest).
- The deployable compact set reaches CV ROC-AUC 0.7254; it deliberately excludes two clinician-scored scales (`FunctionalAssessment` and `ADL`), which lowers accuracy but makes the form answerable by a user.
- Family history showed no positive effect in this data: odds ratio 0.852, chi-square p-value 0.140698, with negative standardized logistic-regression coefficients in all three feature sets.
- Only people aged 60-90 are represented, and the data are cross-sectional, so this estimates current status rather than future risk.
