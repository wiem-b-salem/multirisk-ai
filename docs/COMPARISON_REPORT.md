# Final Comparison Report

## 1. Problem definition and scope

MultiRisk AI estimates the **current** disease risk for cardiovascular disease (CVD), Type 2 diabetes, Alzheimer's disease, and liver disease. The deployed comparison uses the same validated user input for both a traditional machine-learning model and a deep-learning model. The output is an academic estimate, not a medical diagnosis and not a calibrated probability.

## 2. Datasets

| Disease | Source dataset | Rows | Target | Served feature set |
|---------|----------------|------|--------|--------------------|
| CVD | Coronary Heart Disease (`CHDdata.csv`) | 462 | `chd` | 9 features |
| Diabetes | Diabetes Health Dataset Analysis (`diabetes_data.csv`) | 1,879 | `Diagnosis` | 12 compact features |
| Alzheimer's | Kaggle Alzheimer's Disease Dataset (`alzheimers_disease_data.csv`) | 2,149 | `Diagnosis` | 12 compact features |
| Liver | Liver disease dataset (`Liver_disease_data.csv`) | 1,700 | `Diagnosis` | 10 full features |

All four datasets are cross-sectional. The diabetes, Alzheimer's, and liver datasets appear synthetic; several codes and units remain undocumented and are marked for verification on the source pages. The CVD sample is a case-control sample of men aged 15–64 from a high-risk region of South Africa.

## 3. Preprocessing and feature sets

Each experiment uses a stratified 80/20 train/test split with `random_state=42`. A `StandardScaler` is fitted on the training data only. The served feature sets are defined in `models/*/schema_*.json`:

- CVD: 9 clinical and lifestyle features, with `famhist` mapped from `Absent`/`Present` to `0`/`1`.
- Diabetes: 12 compact features, including family history, BMI, glucose markers, blood pressure, hypertension, prediabetes, urination, and thirst.
- Alzheimer's: 12 compact features, including age, education, BMI, family history, comorbidity flags, MMSE, and memory complaints.
- Liver: 10 full features, including alcohol, smoking, genetic risk, physical activity, diabetes, hypertension, and liver function test value.

For deep-learning thresholds, each out-of-fold split fits its own scaler on that fold's training data. The final served scaler is fitted only on the full training split.

## 4. Machine learning models

The deployed ML models were preserved from the original pipeline:

| Disease | Deployed ML model | Artifact |
|---------|-------------------|----------|
| CVD | Logistic Regression | `models/cardiovascular/final_cvd.joblib` |
| Diabetes | Random Forest | `models/diabetes/final_diabetes_compact.joblib` |
| Alzheimer's | Logistic Regression | `models/alzheimers/final_alzheimers_compact.joblib` |
| Liver | Random Forest | `models/liver/final_liver_full.joblib` |

For logistic regression, the API also returns up to five signed standardized contributions. Random Forest returns neutral scores without fabricated feature explanations.

## 5. Deep learning model

The DL model is a Keras Multilayer Perceptron with the same architecture for every disease:

`Input(n_features) -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.2) -> Dense(1, sigmoid)`.

The served MLP files are:

- `models/cardiovascular/mlp_cvd.keras`
- `models/diabetes/mlp_diabetes_compact.keras`
- `models/alzheimers/mlp_alzheimers_compact.keras`
- `models/liver/mlp_liver_full.keras`

Each model uses the same served feature columns and scaler as the corresponding ML model, so the user submits the input only once.

## 6. Experimental setup

`src/deep_serving.py` trains the final MLP on the training split, evaluates it once on the held-out test split, and builds DL risk thresholds from five-fold out-of-fold predictions on the training split only. The test split is never used for model selection, threshold selection, or hyperparameter tuning.

ML thresholds remain in `config/risk_thresholds.json`. DL thresholds are stored separately in `config/dl_risk_thresholds.json`. The threshold rule selects `low_max` as the highest candidate score with recall at least 0.90 and `high_min` as the lowest candidate score with specificity at least 0.90. If those cross, a single two-level threshold is selected by maximizing Youden's J.

## 7. Hyperparameters

| Setting | Value |
|---------|-------|
| Seed | 42 |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Binary crossentropy |
| Metric | AUC |
| Epochs | 200 maximum |
| Batch size | 32 |
| Validation fraction within training data | 0.20 for final model, 0.15 for CV folds |
| Early stopping | Monitor validation loss, patience 15, restore best weights |
| Class weighting | Balanced classes |
| Folds for DL thresholds | 5-fold stratified |
| Dropout | 0.3 then 0.2 |
| Hidden layers | 64 ReLU units then 32 ReLU units |

## 8. Training duration

DL training time was measured with `time.perf_counter()` on the local CPU used for this report. ML training time was not recorded by the existing ML export scripts, so it is reported as not measured rather than estimated.

| Disease | ML training duration | DL training duration |
|---------|----------------------|----------------------|
| CVD | Not measured | 5.48 s |
| Diabetes | Not measured | 10.74 s |
| Alzheimer's | Not measured | 7.51 s |
| Liver | Not measured | 11.29 s |

## 9. Test metrics

Metrics are saved in `results/comparison_metrics.csv` and mirrored for the API in `config/comparison_metrics.json`. Accuracy uses a fixed 0.50 cutoff for both model families.

| Disease | Train / test rows | ML model | ML ROC-AUC | ML accuracy | DL ROC-AUC | DL accuracy | DL OOF ROC-AUC |
|---------|-------------------|----------|------------|-------------|------------|-------------|----------------|
| CVD | 369 / 93 | Logistic Regression | 0.8207 | 0.6774 | 0.7956 | 0.6774 | 0.7358 |
| Diabetes | 1503 / 376 | Random Forest | 0.9547 | 0.9282 | 0.9456 | 0.9016 | 0.9359 |
| Alzheimer's | 1719 / 430 | Logistic Regression | 0.6922 | 0.6163 | 0.7240 | 0.6698 | 0.7213 |
| Liver | 1360 / 340 | Random Forest | 0.9599 | 0.9059 | 0.9152 | 0.8294 | 0.9065 |

Interpretation:

- ML has the higher test ROC-AUC for CVD, diabetes, and liver.
- DL has the higher test ROC-AUC and accuracy for Alzheimer's.
- CVD accuracy is equal for ML and DL.
- These are single held-out test evaluations; they do not establish clinical superiority.

## 10. Deployment and reproducibility

The ML API preserves `POST /predict/{disease}` and adds `POST /compare/{disease}`. The compare response contains `ml`, `dl`, and `comparison` objects. The backend preserves authenticated `POST /predict/{disease}` and adds authenticated `POST /compare/{disease}` plus a new `comparison_predictions` table. `GET /history` returns old ML-only records and new comparison records in one list.

To reproduce locally:

```bash
pip install -r requirements.txt
python -m src.deep_serving
uvicorn src.ml_api.main:app --port 8000
uvicorn src.backend.main:app --port 8001
npm install --prefix frontend
npm run dev --prefix frontend
```

The raw CSV files are gitignored and must be downloaded before training. The served `.keras` files are explicitly unignored in `.gitignore` so deployment can include them. Docker installs TensorFlow through `requirements-ml.txt` and copies `config/` and `models/` into the ML image.

Verification commands used:

```bash
python -m pytest -q
npm run test --prefix frontend
npm run build --prefix frontend
```

## 11. Limitations and responsible use

- All datasets are cross-sectional, so the models estimate current diagnosis risk and not future onset.
- Scores are not calibrated probabilities because training used balanced class weights and, for CVD, a case-control sample.
- Several dataset fields have undocumented units or coding, including LDL, alcohol, adiposity, genetic risk, gender, and ethnicity codes.
- The DL models use the compact or full served feature set only; the non-served research feature sets are not deployed.
- The DL threshold rule optimizes recall and specificity targets on training folds; it does not calibrate probabilities.
- A disagreement between ML and DL is shown neutrally and does not imply that either model is correct.
- This system is for academic demonstration only. Consult a healthcare professional for medical decisions.
