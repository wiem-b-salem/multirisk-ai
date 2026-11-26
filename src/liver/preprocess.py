from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "liver" / "Liver_disease_data.csv"
SEED = 42
FULL_COLUMNS = [
    "Age",
    "Gender",
    "BMI",
    "AlcoholConsumption",
    "Smoking",
    "GeneticRisk",
    "PhysicalActivity",
    "Diabetes",
    "Hypertension",
    "LiverFunctionTest",
]
NO_LAB_COLUMNS = [name for name in FULL_COLUMNS if name != "LiverFunctionTest"]


def load_and_split(feature_set="full", test_size=0.2):
    if feature_set == "full":
        columns = FULL_COLUMNS
    elif feature_set == "no_lab":
        columns = NO_LAB_COLUMNS
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    df = pd.read_csv(DATA_PATH)
    X = df[columns]
    y = df["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=columns, index=X_test.index)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    for feature_set in ("full", "no_lab"):
        print(f"=== {feature_set} ===")
        X_train, X_test, y_train, y_test, _ = load_and_split(feature_set=feature_set)
        print("Train:", X_train.shape, "positive rate:", round(y_train.mean(), 3))
        print("Test: ", X_test.shape, "positive rate:", round(y_test.mean(), 3))
        print("Train mean (should be ~0):", X_train.mean().abs().max().round(6))
