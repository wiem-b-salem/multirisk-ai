from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "diabetes" / "diabetes_data.csv"
SEED = 42
DROP_ALWAYS = ["PatientID", "DoctorInCharge"]
RISK_ONLY_DROP = ["FastingBloodSugar", "HbA1c"]


def load_and_split(feature_set="full", test_size=0.2):
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_ALWAYS)
    df = pd.get_dummies(df, columns=["Ethnicity"], drop_first=True, dtype=int)

    if feature_set == "risk_only":
        df = df.drop(columns=RISK_ONLY_DROP)
    elif feature_set != "full":
        raise ValueError(f"Unknown feature_set: {feature_set}")

    X = df.drop(columns="Diagnosis")
    y = df["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    for feature_set in ("full", "risk_only"):
        print(f"=== {feature_set} ===")
        X_train, X_test, y_train, y_test, _ = load_and_split(feature_set=feature_set)
        print("Train:", X_train.shape, "positive rate:", round(y_train.mean(), 3))
        print("Test: ", X_test.shape, "positive rate:", round(y_test.mean(), 3))
        print("Train mean (should be ~0):", X_train.mean().abs().max().round(6))
