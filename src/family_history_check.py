from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression

from src.cardiovascular.preprocess import load_and_split as load_cvd
from src.diabetes.preprocess import load_and_split as load_diabetes

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def report_association(name, fh, outcome):
    fh = pd.Series(fh).reset_index(drop=True)
    outcome = pd.Series(outcome).reset_index(drop=True)
    pos_absent = int(((fh == 0) & (outcome == 1)).sum())
    neg_absent = int(((fh == 0) & (outcome == 0)).sum())
    pos_present = int(((fh == 1) & (outcome == 1)).sum())
    neg_present = int(((fh == 1) & (outcome == 0)).sum())
    _, p_value, _, _ = chi2_contingency([[pos_absent, neg_absent], [pos_present, neg_present]])
    odds_ratio = (pos_present / neg_present) / (pos_absent / neg_absent)
    print(f"=== {name} ===")
    print(f"Positive rate family history absent:  {pos_absent / (pos_absent + neg_absent):.3f}")
    print(f"Positive rate family history present: {pos_present / (pos_present + neg_present):.3f}")
    print(f"Odds ratio: {odds_ratio:.3f}")
    print(f"Chi-square p-value: {p_value:.6g}")


def logreg_coef(loader, feature):
    X_train, _, y_train, _, _ = loader()
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    model.fit(X_train, y_train)
    return model.coef_[0][list(X_train.columns).index(feature)]


def main():
    cvd = pd.read_csv(ROOT / "data" / "cardiovascular" / "CHDdata.csv")
    cvd_fh = cvd["famhist"].map({"Absent": 0, "Present": 1})
    report_association("CVD (famhist)", cvd_fh, cvd["chd"])

    diabetes = pd.read_csv(ROOT / "data" / "diabetes" / "diabetes_data.csv")
    report_association(
        "Diabetes (FamilyHistoryDiabetes)",
        diabetes["FamilyHistoryDiabetes"],
        diabetes["Diagnosis"],
    )

    print("=== Logistic regression coefficients (standardized) ===")
    print(f"cvd:                {logreg_coef(load_cvd, 'famhist'):.4f}")
    print(f"diabetes_full:      {logreg_coef(lambda: load_diabetes('full'), 'FamilyHistoryDiabetes'):.4f}")
    print(f"diabetes_risk_only: {logreg_coef(lambda: load_diabetes('risk_only'), 'FamilyHistoryDiabetes'):.4f}")


if __name__ == "__main__":
    main()
