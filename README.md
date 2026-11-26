# MultiRisk AI: Comparative Analysis of Machine Learning and Deep Learning Models for Multi-Disease Risk Prediction

MultiRisk AI estimates the **current** risk of four diseases from family history, clinical markers, genetic-risk information, and demographic factors.

The system supports:
1. Cardiovascular Disease
2. Type 2 Diabetes
3. Alzheimer's Disease
4. Liver Disease

For every assessment it compares two independently trained model families:

- **Machine Learning:** Logistic Regression or Random Forest.
- **Deep Learning:** a Neural Network / Multilayer Perceptron (MLP).

Each result shows the actual model name, risk score, risk category, test ROC-AUC, and a short neutral comparison. Inputs are validated once, and history stores only disease, date, ML result, and DL result.

## Key Features

- Schema-driven forms for four diseases.
- Mandatory family-history or genetic-risk input where required.
- Side-by-side ML and DL result cards.
- Separate risk thresholds for ML and DL models.
- Clear risk stratification (low/medium/high or low/high).
- Training-only cross-validation thresholds; test data is used only for final evaluation.
- Backward-compatible history: old ML-only records remain visible.
- Academic disclaimer: this is not a medical diagnosis.

## Note

This MVP uses cross-sectional data to assess **current disease risk**, not future disease onset. Several source datasets are synthetic or have undocumented codes and units; see `docs/METHODOLOGY_RESULTS.md` and `docs/COMPARISON_REPORT.md`.

## Project Structure

- `data/` — Datasets (download required; CSV files are gitignored)
- `src/` — Data pipelines, training scripts, ML API, and backend
- `models/` — Trained ML and MLP artifacts plus served schemas
- `config/` — ML/DL thresholds and comparison metrics
- `results/` — Evaluation CSV files
- `frontend/` — React + Vite app
- `docs/` — Architecture, methodology, and final comparison report
