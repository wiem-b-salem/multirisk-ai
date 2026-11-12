# Phase 2 — Scientific Design

## 1. Problem definition
- Task: binary classification of **current disease risk** (cross-sectional data), per disease.
- Diseases: Type 2 Diabetes, Cardiovascular Disease (Alzheimer's postponed).
- Family history is a mandatory input feature in every model.
- Limitation: no longitudinal follow-up, so this is not future-risk prediction.
- Diabetes dataset is synthetic (Kaggle description); results do not reflect real patients.

## 2. Data
| | Diabetes | CVD |
|---|---|---|
| File | data/diabetes/diabetes_data.csv | data/cardiovascular/CHDdata.csv |
| Rows | 1,879 | 462 |
| Target | Diagnosis | chd |
| Family history | FamilyHistoryDiabetes (0/1) | famhist (Absent/Present) |
| Positive class | 40% | 35% |
| Missing values | 0 | 0 |

## 3. Preprocessing
- Drop identifiers (PatientID, DoctorInCharge).
- Encode famhist: Absent=0, Present=1.
- One-hot encode categorical variables (e.g. Ethnicity).
- Stratified 80/20 train/test split, random seed 42.
- Fit the scaler on the training set only, then apply it to the test set (avoids leakage).
- Class imbalance: class weights.

## 4. Feature sets
- Diabetes full set: all clinical, lifestyle and family-history features.
- Diabetes risk-only set: exclude FastingBloodSugar and HbA1c, since they are near-diagnostic and may dominate the model.
- Compare both sets and report the difference.
- CVD: all 9 features.

## 5. Models
- Baselines: Logistic Regression, Random Forest.
- Deep learning: MLP (2-3 dense layers, ReLU, dropout, sigmoid output, Adam, early stopping).
- CVD has only 462 rows, so the MLP may not beat the baselines. Report honestly.

## 6. Evaluation
- Metrics: ROC-AUC, PR-AUC, precision, recall, specificity, F1, confusion matrix, calibration.
- 5-fold stratified cross-validation on the training set; final metrics on the held-out test set once.
- No tuning on the test set.

## 7. Risk output (hypothesis to validate)
- Output: probability, risk level (low/medium/high), top contributing factors, disclaimer.
- Thresholds to be chosen from the calibration and ROC analysis, not fixed in advance.

## 8. Open decisions
- Separate per-disease models vs one multi-task model (current choice: separate).
- Alzheimer's dataset.
- Professor feedback on cross-sectional and synthetic data.