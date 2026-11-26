# MultiRisk AI Explained

MultiRisk AI is an academic web application that estimates the **current** risk of four diseases from entered health features: cardiovascular disease, type 2 diabetes, Alzheimer's disease, and liver disease. It is not a medical diagnosis, not a clinical decision-support tool, and not a calibrated probability estimate.

This document supersedes older project notes. Earlier text saying there was **no implemented MLP/DL model** is stale: the current application serves one Keras MLP per disease and exposes a dedicated ML-versus-DL comparison endpoint.

## 1. Project scope

The application does four things:

1. Loads a disease-specific schema from disk.
2. Validates user-entered raw fields and converts them into model features.
3. Computes an independent classical machine-learning score and a deep-learning/MLP score.
4. Shows both scores with risk categories, test metrics, and a neutral comparison note.

The served API disease keys are:

| API key | Disease label used in the UI |
|---|---|
| `cvd` | Cardiovascular |
| `diabetes` | Type 2 Diabetes |
| `alzheimers` | Alzheimer's disease |
| `liver` | Liver disease |

The application estimates current diagnosis risk from cross-sectional data. It does not predict future disease onset.

## 2. Final architecture

### 2.1 Runtime components

```text
Browser
  |
  | HTTPS/HTTP REST + JSON
  v
React SPA served by nginx (frontend)
  |
  | REST + JSON, Bearer JWT
  v
FastAPI backend (auth, history, profiles, ML API proxy)
  |
  | internal HTTP only
  v
FastAPI ML API (schemas, validation, ML + MLP inference)
  |
  v
Model/schema/config files on disk

Backend storage:
  - SQLite in local development: sqlite:///./backend.db
  - PostgreSQL 16 in Docker
```

Only the frontend and backend are published to the host in Docker. The ML API and database stay on the internal Docker network. In local development, the documented commands start the ML API on `http://127.0.0.1:8000`, the backend on `http://127.0.0.1:8001`, and the Vite frontend through `npm run dev --prefix frontend`. The exact Vite dev-server port is unclear from the code; the frontend default API URL is `http://127.0.0.1:8001`.

AWS deployment is planned future work and is not implemented. There are no AWS, Terraform, or cloud deployment files in the repository.

### 2.2 Docker services and ports

| Service | Image/Dockerfile | Internal port | Host port | Purpose |
|---|---|---:|---:|---|
| `ml_api` | `docker/ml.Dockerfile` | `8000` | none | Model inference and schemas; healthcheck on `/health` |
| `db` | `postgres:16-alpine` | `5432` | none | PostgreSQL storage with named volume `db_data` |
| `backend` | `docker/backend.Dockerfile` | `8001` | `8001` | Auth, profiles, history, ML API proxy |
| `frontend` | `docker/frontend.Dockerfile` | `80` | `${FRONTEND_PORT:-8080}` | nginx-served React SPA |

The frontend image builds the SPA with `VITE_API_URL`, defaulting to `http://localhost:8001`. Because Vite embeds this at build time, runtime-only changes to the built SPA are unclear from the code.

### 2.3 API routes

ML API, `src/ml_api/main.py`:

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Returns `ok` and the four disease keys |
| `GET` | `/schema/{disease}` | none | Returns the served schema |
| `POST` | `/predict/{disease}` | none | Classical ML-only prediction; kept for backward compatibility |
| `POST` | `/compare/{disease}` | none | ML result, DL result, and comparison summary |

Backend, `src/backend/main.py`:

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Backend healthcheck |
| `POST` | `/auth/register` | none | Create user |
| `POST` | `/auth/login` | none | Issue Bearer JWT |
| `GET` | `/me` | Bearer | Current user |
| `GET` | `/schema/{disease}` | none | Proxy ML API schema |
| `POST` | `/predict/{disease}` | Bearer | Proxy `/predict` and store ML-only history |
| `POST` | `/compare/{disease}` | Bearer | Proxy `/compare` and store ML/DL history |
| `GET` | `/history` | Bearer | Merge old ML-only and new comparison records |
| `GET` | `/profile` | Bearer | Read decrypted profile data |
| `PUT` | `/profile` | Bearer | Validate and store encrypted profile data |
| `DELETE` | `/profile` | Bearer | Delete encrypted profile data |

Auth details:

- Passwords are hashed with bcrypt.
- JWTs use HS256 with a 60-minute expiry.
- The frontend stores the token in `sessionStorage`.
- Logout is client-side only; there is no refresh, revocation, or password-reset endpoint.
- Production startup requires a non-default `JWT_SECRET` of at least 32 characters and a valid `PROFILE_ENCRYPTION_KEY`.

### 2.4 Privacy and storage

The backend tables are:

| Table | Stored data |
|---|---|
| `users` | `id`, unique `email`, `password_hash`, `created_at` |
| `predictions` | Old ML-only records: `id`, `user_id`, `disease`, `risk_level`, `risk_score`, `created_at` |
| `comparison_predictions` | `id`, `user_id`, `disease`, `ml_risk_level`, `ml_risk_score`, `dl_risk_level`, `dl_risk_score`, `created_at` |
| `profiles` | `id`, unique `user_id`, Fernet-encrypted `data_encrypted`, `updated_at` |

Raw health inputs are submitted once and are not stored in prediction or comparison history. The smoke test checks that no history key contains `feature`.

Profile data is restricted to keys matching `^(shared|cvd|diabetes|alzheimers|liver|smoking)\.[A-Za-z0-9_]{1,40}$`, at most 100 keys, serialized values of at most 10,240 bytes, and scalar or short string values only.

## 3. Implemented models

### 3.1 Model families

Three model families were evaluated:

| Family | Implementation | Hyperparameters |
|---|---|---|
| Logistic regression | `src/baselines.py` | `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)` |
| Random forest | `src/baselines.py` | `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)` |
| MLP neural network | `src/mlp.py`, disease MLP scripts, `src/deep_serving.py` | `Dense(64, relu)`, `Dropout(0.3)`, `Dense(32, relu)`, `Dropout(0.2)`, `Dense(1, sigmoid)`; Adam `0.001`, binary cross-entropy, AUC metric |

The MLP is a small dense feed-forward neural network. It is not a clinical image, text, or time-series model.

### 3.2 Served model matrix

| Disease | API setup | Classical ML model | DL/MLP model | Served features |
|---|---|---|---|---:|
| `cvd` | `cvd` | `final_cvd.joblib` (`logreg`) | `mlp_cvd.keras` | 9 |
| `diabetes` | `diabetes_compact` | `final_diabetes_compact.joblib` (`rf`) | `mlp_diabetes_compact.keras` | 12 |
| `alzheimers` | `alzheimers_compact` | `final_alzheimers_compact.joblib` (`logreg`) | `mlp_alzheimers_compact.keras` | 12 |
| `liver` | `liver_full` | `final_liver_full.joblib` (`rf`) | `mlp_liver_full.keras` | 10 |

Served artifacts:

| Disease | Schema | ML artifact | Scaler | DL artifact |
|---|---|---|---|---|
| `cvd` | `models/cardiovascular/schema_cvd.json` | `models/cardiovascular/final_cvd.joblib` | `models/cardiovascular/scaler_cvd.joblib` | `models/cardiovascular/mlp_cvd.keras` |
| `diabetes` | `models/diabetes/schema_diabetes_compact.json` | `models/diabetes/final_diabetes_compact.joblib` | `models/diabetes/scaler_diabetes_compact.joblib` | `models/diabetes/mlp_diabetes_compact.keras` |
| `alzheimers` | `models/alzheimers/schema_alzheimers_compact.json` | `models/alzheimers/final_alzheimers_compact.joblib` | `models/alzheimers/scaler_alzheimers_compact.joblib` | `models/alzheimers/mlp_alzheimers_compact.keras` |
| `liver` | `models/liver/schema_liver_full.json` | `models/liver/final_liver_full.joblib` | `models/liver/scaler_liver_full.joblib` | `models/liver/mlp_liver_full.keras` |

Research-only artifacts also exist, such as diabetes `full`/`risk_only` models, Alzheimer's full/risk-only scalers, and liver no-lab scalers. They are not loaded by the ML API.

### 3.3 Why ML and DL are separate outputs

The classical ML model remains the primary deployed result for backward compatibility. The MLP is served beside it so the UI can show how two different model families respond to the same submitted features.

The application does **not** claim that either model is clinically correct when they disagree. Disagreement is presented neutrally.

## 4. Data and preprocessing

### 4.1 Datasets and splits

All experiments use a stratified 80/20 train/test split with `random_state=42`.

| Disease | CSV | Full shape | Target | Class counts | Train rows | Test rows | Served features |
|---|---|---:|---|---|---:|---:|---:|
| CVD | `data/cardiovascular/CHDdata.csv` | `(462, 10)` | `chd` | `{0: 302, 1: 160}` | 369 | 93 | 9 |
| Diabetes | `data/diabetes/diabetes_data.csv` | `(1879, 46)` | `Diagnosis` | `{0: 1127, 1: 752}` | 1503 | 376 | 12 |
| Alzheimer's | `data/alzheimers/alzheimers_disease_data.csv` | `(2149, 35)` | `Diagnosis` | `{0: 1389, 1: 760}` | 1719 | 430 | 12 |
| Liver | `data/liver/Liver_disease_data.csv` | `(1700, 11)` | `Diagnosis` | `{0: 764, 1: 936}` | 1360 | 340 | 10 |

Raw CSVs are gitignored by `data/**/*.csv`. A fresh clone therefore needs the datasets downloaded before retraining. The served artifacts already in `models/` and `config/` are used at inference time.

### 4.2 Served feature columns

CVD:

```text
sbp
tobacco
ldl
adiposity
famhist
typea
obesity
alcohol
age
```

Diabetes compact:

```text
Age
Gender
BMI
FamilyHistoryDiabetes
FastingBloodSugar
HbA1c
SystolicBP
DiastolicBP
Hypertension
PreviousPreDiabetes
FrequentUrination
ExcessiveThirst
```

Alzheimer's compact:

```text
Age
Gender
EducationLevel
BMI
FamilyHistoryAlzheimers
CardiovascularDisease
Diabetes
Depression
HeadInjury
Hypertension
MMSE
MemoryComplaints
```

Liver full:

```text
Age
Gender
BMI
AlcoholConsumption
Smoking
GeneticRisk
PhysicalActivity
Diabetes
Hypertension
LiverFunctionTest
```

The schemas contain the full raw-field definitions, labels, units, help text, training ranges, validation ranges, and allowed categorical values.

### 4.3 Disease-specific preprocessing

Common steps:

- Stratified train/test split with `random_state=42`.
- `StandardScaler` fitted on training rows only.
- Test rows transformed with the fitted scaler.
- Served scalers fitted on the training split and stored beside served models.
- For DL out-of-fold thresholding, each fold fits a fresh scaler on that fold's training rows only.

Disease-specific behavior:

| Disease | Preprocessing |
|---|---|
| CVD | Maps `famhist` from `Absent`/`Present` to `0`/`1`; drops target `chd`; scales the nine served columns. |
| Diabetes | Drops `PatientID` and `DoctorInCharge`. `full` and `risk_only` one-hot encode `Ethnicity` with `drop_first=True`. `risk_only` drops `FastingBloodSugar` and `HbA1c`. Served `compact` uses the fixed 12 columns above. |
| Alzheimer's | Drops `PatientID` and `DoctorInCharge`. `full` and `risk_only` one-hot encode `Ethnicity` with `drop_first=True`. `risk_only` drops cognitive/functional fields. Served `compact` uses the fixed 12 columns above and excludes clinician-scored `FunctionalAssessment` and `ADL`. |
| Liver | Uses fixed `FULL_COLUMNS` and `NO_LAB_COLUMNS`. Served setup is `full`. `GeneticRisk` remains numeric `0/1/2`; it is a genetic-risk level, not a family-history question. |

### 4.4 Frontend-derived inputs

The frontend derives some raw values before submission:

- BMI: `round(weightKg / (heightCm / 100)^2, 1)`.
- CVD `tobacco`: cumulative lifetime kilograms from cigarettes per day, years smoked, and `GRAMS_PER_CIGARETTE = 1`.

Validation ranges:

- Weight: 25–300 kg.
- Height: 100–230 cm.
- Cigarettes per day: 0–100.
- Years smoked: 0–80.

### 4.5 API validation and soft warnings

The ML API builds a Pydantic request model from the schema:

- Every `raw_field` is required.
- `allowed_values` become `Literal[...]`.
- `valid_min` and `valid_max` are hard bounds; violations return `422`.
- `train_min` and `train_max` are soft bounds. Values outside them are accepted but add `out_of_training_range` and `range_warning`.
- CVD `famhist` is mapped from `Absent`/`Present` to `0`/`1`.
- The request frame is reindexed to `schema["feature_columns"]` with `fill_value=0`, then passed through the saved scaler.

## 5. Training

### 5.1 Classical ML training

Shared model definitions and evaluation live in `src/baselines.py`.

Protocol:

1. Build model candidates: logistic regression and random forest.
2. Run `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on training data.
3. Score with ROC-AUC using `cross_val_score`.
4. Select the model with the highest `cv_roc_auc_mean`.
5. Fit the selected model on the full training split.
6. Save joblib model and scaler.
7. Evaluate the held-out test split once.

Final served ML selections:

| Disease | Served setup | Winner | CV ROC-AUC mean | CV ROC-AUC std | Runner-up CV ROC-AUC |
|---|---|---|---:|---:|---:|
| CVD | `cvd` | `logreg` | `0.7590979853479853` | `0.06502506968473151` | `rf`: `0.713486002093145` |
| Diabetes | `diabetes_compact` | `rf` | `0.957416051313402` | `0.01185623113175325` | `logreg`: `0.903280405765338` |
| Alzheimer's | `alzheimers_compact` | `logreg` | `0.7254457210249048` | `0.006476633140227928` | `rf`: `0.7243127549313351`; MLP: `0.7217389110467509` |
| Liver | `liver_full` | `rf` | `0.9370158191756854` | `0.011624716137325375` | `logreg`: `0.903741044969037`; MLP: `0.9082643953604241` |

### 5.2 MLP training

The served MLP flow is implemented in `src/deep_serving.py`.

Final served model:

- Reads raw CSV.
- Selects `schema["feature_columns"]`.
- Maps CVD `famhist` if present.
- Uses the same stratified 80/20 split with seed 42.
- Fits a final scaler on training rows.
- Builds the model with seed 42.
- Trains with an internal validation split of 20% of training rows.
- Saves `.keras` and overwrites the served scaler file.
- Evaluates once on the held-out test split.

Training parameters:

```text
optimizer: Adam(learning_rate=0.001)
loss: binary_crossentropy
metrics: AUC
epochs: 200
batch_size: 32
validation_split for final model: 0.2
validation_split for OOF folds: 0.15
early stopping: val_loss, patience=15, restore_best_weights=True
class weights: balanced
seed: 42 for served models
```

Research MLP scripts also train seeds `0-4` and report `test_roc_auc_seed_mean` and `test_roc_auc_seed_std`. Served MLPs use seed 42.

### 5.3 DL thresholds

DL thresholds are derived from five-fold out-of-fold training predictions, never from the test split:

1. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
2. Each fold fits a fresh scaler on that fold's training rows.
3. Each fold trains a fresh MLP with seed 42.
4. Held-out fold predictions fill an out-of-fold array.
5. `cv_roc_auc_mean` is pooled `roc_auc_score(y_train, oof)`, not a mean of fold AUCs.
6. `choose_thresholds` selects `low_max` and `high_min` from unique OOF scores.

Threshold rule:

- `low_max`: highest candidate with recall at least `0.90`; fallback to the minimum candidate.
- `high_min`: lowest candidate with specificity at least `0.90`; fallback to the maximum candidate.
- If they cross, a single Youden's J-style threshold is used and both values are set equal.

ML thresholds are stored separately in `config/risk_thresholds.json`; DL thresholds are stored in `config/dl_risk_thresholds.json`.

### 5.4 Training outputs

`src/deep_serving.py` writes:

- `models/cardiovascular/mlp_cvd.keras`
- `models/diabetes/mlp_diabetes_compact.keras`
- `models/alzheimers/mlp_alzheimers_compact.keras`
- `models/liver/mlp_liver_full.keras`
- Served scaler joblib files
- `config/dl_risk_thresholds.json`
- `config/comparison_metrics.json`
- `results/comparison_metrics.csv`

It does not retrain the classical ML models. Run the classical retraining scripts first if both families need to be regenerated.

## 6. Prediction flow

### 6.1 One submission, two predictions

The frontend submits one raw-feature object per disease to `POST /compare/{disease}` through the backend. The backend authenticates the user, forwards the request to the ML API, stores the result summary, and returns it to the frontend.

The ML API:

1. Resolves the disease schema.
2. Validates all raw fields.
3. Builds and scales one feature frame.
4. Loads the classical model, scaler, MLP, thresholds, and metrics.
5. Computes the ML score with `predict_proba[:, 1]`.
6. Computes the DL score with sigmoid model output.
7. Converts each score to a risk level using that family's thresholds.
8. Adds notes, warnings, metrics, and the comparison summary.

The older `POST /predict/{disease}` endpoint remains available and returns only the ML result.

### 6.2 Risk levels

For thresholds where `low_max == high_min`:

- `score <= low_max` → `low`
- otherwise → `high`

For three-band thresholds:

- `score < low_max` → `low`
- `score >= high_min` → `high`
- otherwise → `medium`

ML thresholds:

| Disease | `low_max` | `high_min` | Bands |
|---|---:|---:|---|
| `cvd` | `0.27703360102750624` | `0.711124446390867` | low / medium / high |
| `diabetes` | `0.44` | `0.44` | low / high |
| `alzheimers` | `0.2985038875290284` | `0.6448177527418835` | low / medium / high |
| `liver` | `0.47333333333333333` | `0.54` | low / medium / high |

DL thresholds:

| Disease | `low_max` | `high_min` | Bands |
|---|---:|---:|---|
| `cvd` | `0.3205692172050476` | `0.7150633335113525` | low / medium / high |
| `diabetes` | `0.404654324054718` | `0.530335545539856` | low / medium / high |
| `alzheimers` | `0.2949957847595215` | `0.653008759021759` | low / medium / high |
| `liver` | `0.3442036807537079` | `0.6762911677360535` | low / medium / high |

Diabetes ML uses two levels; diabetes DL uses three levels because its separate OOF thresholds do not collapse to one value.

### 6.3 Result objects

The ML result and the `ml` object in `/compare` contain:

- `disease`
- `model_name`
- `risk_score`
- `risk_level`
- `score_note`
- `top_factors`
- `disclaimer`
- `test_roc_auc`
- `test_accuracy`
- `training_seconds`
- `population_note`

The DL `dl` object contains the same essentials plus:

- `model_label: "Neural Network (MLP)"`
- `architecture: "Dense(64) - Dropout(0.3) - Dense(32) - Dropout(0.2)"`
- `top_factors: null`

Conditional response fields:

- Logistic-regression models return up to five `top_factors`.
- Random-forest models return `top_factors: null`.
- Diabetes, Alzheimer's, and liver results include `model_note`.
- CVD and Alzheimer's logistic-regression results include `factors_note`.
- Out-of-training-range submissions include `out_of_training_range` and `range_warning`.

`top_factors` are signed standardized contributions from logistic regression: coefficient multiplied by scaled feature value, sorted by absolute contribution. They are relative to the average training person, not causal explanations.

### 6.4 Medical notes

The API uses these exact notes:

```text
This is an estimate for academic demonstration, not a medical diagnosis. Consult a healthcare professional.

Model score, not a calibrated probability: models were trained with balanced class weights.

Neural network score, not a calibrated probability: the MLP was trained with balanced class weights and uses separate DL thresholds.

Different model architectures can disagree; disagreement is not a medical diagnosis and should be interpreted with caution.

Contributions are relative to the average person in the training data.

One or more values are outside the range the model was trained on; the estimate may be unreliable.
```

## 7. ML-versus-DL comparison flow

### 7.1 API comparison object

`POST /compare/{disease}` returns:

```json
{
  "disease": "cvd",
  "ml": {},
  "dl": {},
  "comparison": {
    "score_difference": 0.0,
    "risk_levels_match": true,
    "interpretation": "Both models classify this input as low risk.",
    "note": "Different model architectures can disagree; disagreement is not a medical diagnosis and should be interpreted with caution.",
    "ml_test_roc_auc": 0.0,
    "dl_test_roc_auc": 0.0,
    "ml_test_accuracy": 0.0,
    "dl_test_accuracy": 0.0
  },
  "disclaimer": "..."
}
```

`score_difference` is signed as `dl_score - ml_score`. The frontend displays its absolute value.

Interpretation strings:

- Same level: `"Both models classify this input as {level} risk."`
- Different levels: `"ML classifies this input as {ml_level} risk, while the neural network classifies it as {dl_level} risk."`

### 7.2 Frontend rendering

`ComparisonResult.jsx` renders two cards:

- `Machine Learning — Logistic Regression` or `Machine Learning — Random Forest`
- `Deep Learning — MLP Neural Network`

Each card shows the risk category, score out of 100, score note, and test ROC-AUC when available. Below the cards, the UI shows absolute score difference, interpretation, and the comparison note.

`Predict.jsx` uses the comparison endpoint for one disease. `Assessment.jsx` loads all four schemas, lets the user select one or more diseases, shares common fields where possible, and sends one comparison request per selected disease.

`History.jsx` displays `Date`, `Disease`, `ML result`, and `DL result`. Old ML-only records are shown with `—` in the DL column.

### 7.3 History records

Comparison rows store only:

- `ml_risk_level`
- `ml_risk_score`
- `dl_risk_level`
- `dl_risk_score`
- disease, user, and timestamp

There is no raw-feature storage in either old or new history records.

## 8. Evaluation metrics and actual results

### 8.1 Metric definitions

From `src/baselines.py` and reused by DL evaluation:

```text
predicted class: score >= 0.5
TN, FP, FN, TP from confusion_matrix
ROC-AUC: roc_auc_score(y_true, scores)
PR-AUC: average_precision_score(y_true, scores)
precision: precision_score(y_true, predicted class)
recall: recall_score(y_true, predicted class)
specificity: TN / (TN + FP)
F1: f1_score(y_true, predicted class)
accuracy: accuracy_score(y_true, predicted class)
```

ML CV AUC uses `cross_val_score(..., scoring="roc_auc")` on training rows. DL CV AUC in the served comparison uses pooled out-of-fold training predictions. The held-out test split is evaluated once.

Scores are not calibrated probabilities because class weights were balanced.

### 8.2 Served comparison results

Source: `results/comparison_metrics.csv`, mirrored for the API in `config/comparison_metrics.json`.

Discrimination and accuracy:

| Disease | Family | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test accuracy | Training seconds |
|---|---|---:|---:|---:|---:|---:|
| CVD | ML logreg | not recorded in comparison file | `0.8206967213114754` | `0.7140637389932847` | `0.6774193548387096` | not recorded |
| CVD | DL MLP | `0.735833765560166` | `0.795594262295082` | `0.6304221626784207` | `0.6774193548387096` | `5.4771757000125945` |
| Diabetes | ML rf | not recorded in comparison file | `0.9547345132743362` | `0.9400604182223624` | `0.9281914893617021` | not recorded |
| Diabetes | DL MLP | `0.935864912002537` | `0.9456047197640118` | `0.9008401576250521` | `0.901595744680851` | `10.737579999957234` |
| Alzheimer's | ML logreg | not recorded in comparison file | `0.6921857251041272` | `0.5908730217676162` | `0.6162790697674418` | not recorded |
| Alzheimer's | DL MLP | `0.7213451608318726` | `0.7239681938659599` | `0.6029772312488426` | `0.6697674418604651` | `7.506818299996667` |
| Liver | ML rf | not recorded in comparison file | `0.9598755723323197` | `0.9542538205687591` | `0.9058823529411765` | not recorded |
| Liver | DL MLP | `0.9065486114601248` | `0.9152423892908321` | `0.9275711002029251` | `0.8294117647058824` | `11.290463599958457` |

Classification metrics at threshold 0.5:

| Disease | Family | Precision | Recall | Specificity | F1 | TN / FP / FN / TP |
|---|---|---:|---:|---:|---:|---:|
| CVD | ML logreg | `0.5185185185185185` | `0.875` | `0.5737704918032787` | `0.6511627906976745` | `35 / 26 / 4 / 28` |
| CVD | DL MLP | `0.52` | `0.8125` | `0.6065573770491803` | `0.6341463414634146` | `37 / 24 / 6 / 26` |
| Diabetes | ML rf | `0.9300699300699301` | `0.8866666666666667` | `0.9557522123893806` | `0.9078498293515358` | `216 / 10 / 17 / 133` |
| Diabetes | DL MLP | `0.8598726114649682` | `0.9` | `0.9026548672566371` | `0.8794788273615635` | `204 / 22 / 15 / 135` |
| Alzheimer's | ML logreg | `0.4644808743169399` | `0.5592105263157895` | `0.6474820143884892` | `0.5074626865671642` | `180 / 98 / 67 / 85` |
| Alzheimer's | DL MLP | `0.532051282051282` | `0.5460526315789473` | `0.737410071942446` | `0.538961038961039` | `205 / 73 / 69 / 83` |
| Liver | ML rf | `0.9378531073446328` | `0.8877005347593583` | `0.9281045751633987` | `0.9120879120879121` | `142 / 11 / 21 / 166` |
| Liver | DL MLP | `0.8771929824561403` | `0.8021390374331551` | `0.8627450980392157` | `0.8379888268156425` | `132 / 21 / 37 / 150` |

Interpretation:

- Classical ML has higher held-out test ROC-AUC for CVD, diabetes, and liver.
- DL has higher held-out test ROC-AUC and accuracy for Alzheimer's.
- CVD accuracy is equal for both families.
- These are single held-out evaluations and do not establish clinical superiority.

## 9. Which files implement each part

| Part | Files |
|---|---|
| Project overview | `README.md`, `PROJECT_EXPLAINED.md` |
| Final architecture and verification | `docs/ARCHITECTURE.md` |
| Comparison methodology and results | `docs/COMPARISON_REPORT.md`, `docs/METHODOLOGY_RESULTS.md` |
| Dataset notes | `docs/PHASE1_DATASET_RESEARCH.md`, `docs/ALZHEIMERS_RESULTS.md`, `docs/LIVER_RESULTS.md` |
| Shared ML models and metrics | `src/baselines.py` |
| Shared MLP architecture | `src/mlp.py` |
| CVD research pipeline | `src/cardiovascular/preprocess.py`, `src/baselines.py`, `src/mlp.py`, `src/report.py`, `src/calibrate_thresholds.py`, `src/export_artifacts.py` |
| Diabetes research and served compact pipeline | `src/diabetes/preprocess.py`, `src/diabetes_compact.py`, `src/baselines.py`, `src/mlp.py`, `src/calibrate_thresholds.py`, `src/export_artifacts.py` |
| Alzheimer's pipeline | `src/alzheimers/preprocess.py`, `src/alzheimers/baselines.py`, `src/alzheimers/mlp.py`, `src/alzheimers/report.py`, `src/alzheimers/calibrate_thresholds.py`, `src/alzheimers/export_artifacts.py` |
| Liver pipeline | `src/liver/preprocess.py`, `src/liver/baselines.py`, `src/liver/mlp.py`, `src/liver/report.py`, `src/liver/calibrate_thresholds.py`, `src/liver/export_artifacts.py` |
| Served DL training and comparison outputs | `src/deep_serving.py` |
| Served schemas | `models/cardiovascular/schema_cvd.json`, `models/diabetes/schema_diabetes_compact.json`, `models/alzheimers/schema_alzheimers_compact.json`, `models/liver/schema_liver_full.json` |
| ML thresholds | `config/risk_thresholds.json`, `src/calibrate_thresholds.py`, `src/alzheimers/calibrate_thresholds.py`, `src/liver/calibrate_thresholds.py` |
| DL thresholds and API metrics | `config/dl_risk_thresholds.json`, `config/comparison_metrics.json`, `results/comparison_metrics.csv` |
| Field help text | `config/field_help.json` |
| Inference API | `src/ml_api/main.py` |
| Auth, profiles, proxy, history | `src/backend/main.py` |
| Frontend API base and tokens | `frontend/src/api.js` |
| Frontend routes | `frontend/src/App.jsx` |
| Login and register | `frontend/src/pages/Login.jsx`, `frontend/src/pages/Register.jsx` |
| Single-disease assessment | `frontend/src/pages/Predict.jsx` |
| Multi-disease assessment | `frontend/src/pages/Assessment.jsx` |
| History table | `frontend/src/pages/History.jsx` |
| Comparison cards | `frontend/src/components/ComparisonResult.jsx` |
| Single result card | `frontend/src/components/ResultCard.jsx` |
| Dynamic form rendering | `frontend/src/components/FieldInput.jsx`, `frontend/src/sharedFields.js`, `frontend/src/uiCopy.js`, `frontend/src/labels.js` |
| Derived inputs | `frontend/src/derived.js` |
| Docker stack | `docker-compose.yml`, `docker/ml.Dockerfile`, `docker/backend.Dockerfile`, `docker/frontend.Dockerfile`, `docker/nginx.conf` |
| Environment template | `.env.example` |
| Smoke test | `scripts/smoke_test.py` |
| CI | `.github/workflows/ci.yml` |
| Python tests | `tests/test_ml_api.py`, `tests/test_backend.py`, `tests/test_backend_profile.py` |
| Frontend tests | `frontend/src/pages/Predict.test.jsx`, `frontend/src/pages/Assessment.test.jsx`, and neighboring Vitest tests |
| Dependency sets | `requirements.txt`, `requirements-ml.txt`, `requirements-backend.txt`, `requirements-dev.txt`, `frontend/package.json` |
| Ignore rules | `.gitignore`, `.dockerignore` |

Human/team ownership is unclear from the code. No `CODEOWNERS`, `OWNERS*`, or `AUTHORS*` files were found.

## 10. How to run

### 10.1 Prerequisites

- Python 3.13-compatible environment.
- Node 20-compatible environment for frontend build/tests.
- Docker and Docker Compose for the containerized stack.
- Raw CSV datasets under `data/` only if retraining from scratch.

Python dependencies are split by role:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-ml.txt -r requirements-backend.txt -r requirements-dev.txt
```

`requirements.txt` is a broad combined development list. TensorFlow is required by `requirements-ml.txt` for the served MLPs.

### 10.2 Local serving

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
npm install --prefix frontend
```

Regenerate DL artifacts and comparison metrics if needed:

```powershell
.\venv\Scripts\python.exe -m src.deep_serving
```

Start the services in separate terminals:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.ml_api.main:app --port 8000
.\venv\Scripts\python.exe -m uvicorn src.backend.main:app --port 8001
npm run dev --prefix frontend
```

The backend defaults:

```text
DATABASE_URL=sqlite:///./backend.db
ML_API_URL=http://127.0.0.1:8000
JWT_SECRET=dev-secret-change-me
ENVIRONMENT=development
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

For production-like local runs, set a strong `JWT_SECRET`, a valid Fernet `PROFILE_ENCRYPTION_KEY`, and suitable `CORS_ORIGINS`.

### 10.3 Full retraining

Raw CSVs must be present. The documented PowerShell sequence is:

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
.\venv\Scripts\python.exe -m src.deep_serving
```

The current `src.calibrate_thresholds.py` preserves existing Alzheimer's and liver keys rather than dropping them.

### 10.4 Docker stack

Copy `.env.example` to `.env`, then set:

```text
POSTGRES_PASSWORD
JWT_SECRET
PROFILE_ENCRYPTION_KEY
FRONTEND_PORT
VITE_API_URL
```

Generate a Fernet key with:

```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Start and verify:

```bash
docker compose config --quiet
docker compose up --build -d --wait
python scripts/smoke_test.py
docker compose logs --no-color
docker compose down -v
```

Required Compose variables are `POSTGRES_PASSWORD`, `JWT_SECRET`, and `PROFILE_ENCRYPTION_KEY`.

If host port 8080 is occupied, set a different frontend port and point the smoke test at it. On PowerShell:

```powershell
$env:FRONTEND_PORT='18080'
$env:FRONTEND_URL='http://localhost:18080'
```

Backend remains published on host port 8001.

### 10.5 Tests and verification

Documented latest results:

| Check | Command | Result |
|---|---|---|
| Python tests | `.\\venv\\Scripts\\python.exe -m pytest -q` | `69 passed, 1 warning in 20.09s` |
| Frontend tests | `npm run test` | Vitest `v2.1.9`, `8` files and `36` tests passed |
| Frontend build | `npm run build` | Vite transformed `49` modules and built successfully in `1.20s` |
| Compose validation | `docker compose config --quiet` | passed with required environment variables set |
| Docker smoke | `python scripts/smoke_test.py` | passed with required environment variables set |

The smoke test checks backend health, registration, login, all four schemas, all four compare endpoints, incomplete-payload rejection, four saved comparison records, absence of feature keys in history, profile create/read/delete, the frontend root, and SPA fallback on `/history`.

## 11. Limitations and caveats

### 11.1 Medical and statistical scope

- All datasets are cross-sectional; the models estimate current diagnosis status, not future onset.
- Scores are not calibrated probabilities because class weights were balanced.
- CVD is a case-control sample of men aged 15–64 from a high-risk region of South Africa. It is not validated for women or people outside that age range.
- Diabetes, Alzheimer's, and liver datasets appear synthetic and are not validated on real patients.
- ML/DL disagreement is not a medical diagnosis and does not imply either model is correct.
- The system is for academic demonstration only.

### 11.2 Data interpretation

Use `unclear from the code` for undocumented meanings and units.

CVD:

- `ldl` unit: unclear from the code.
- `adiposity` definition: unclear from the code.
- `alcohol` unit: unclear from the code.
- `tobacco` is documented as cumulative lifetime kilograms; the frontend derives it from smoking behavior.

Diabetes:

- `Gender` 0/1 meaning: unclear from the code.
- Kaggle URL is documented as `https://www.kaggle.com/datasets/rabieelkharoua/diabetes-health-dataset-analysis`.
- The research note reports Kaggle CC BY 4.0, but this must be verified on the source page.
- Performance depends heavily on `FastingBloodSugar` and `HbA1c`; removing them drops test AUC sharply.

Alzheimer's:

- Exact Kaggle URL: unclear from the code.
- Licence and provenance: `to be verified on the source page`.
- `Gender` 0/1 meaning: unclear from the code.
- `EducationLevel` 0–3 meaning: unclear from the code.
- Family history did not show a positive effect in the repository analysis.
- The full random forest is stronger than the served compact logistic regression; the compact set was chosen for a simpler served form.

Liver:

- Exact Kaggle URL: unclear from the code.
- Licence and provenance: `to be verified on the source page`.
- `GeneticRisk` is a level from 0 to 2, not a yes/no family-history field. How it was determined is unclear from the code.
- `Gender` coding, `AlcoholConsumption` unit, `PhysicalActivity` unit, and `LiverFunctionTest` identity/unit are unclear from the code.

CVD exact licence text is unclear from the code and must be verified on the source page.

### 11.3 Current documentation drift

- `docs/PHASE1_DATASET_RESEARCH.md` and `docs/PHASE2_SCIENTIFIC_DESIGN.md` reflect the original two-disease scope and are stale relative to the four-disease deployment.
- `docs/METHODOLOGY_RESULTS.md` still says `src.calibrate_thresholds` drops Alzheimer's and liver thresholds; the current code preserves them.
- `docs/ARCHITECTURE.md` under-describes `/profile` and does not list the `profiles` table in all summaries.
- `docs/ARCHITECTURE.md` says the population note is displayed above the form, but current predict/assessment components do not render `populationNote`.
- `docs/ALZHEIMERS_RESULTS.md` and `docs/LIVER_RESULTS.md` should be checked for stale claims that no served MLP artifacts exist.

This `PROJECT_EXPLAINED.md` is the current high-level source of truth for the implemented comparison stack.
