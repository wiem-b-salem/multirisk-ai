from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "alzheimers" / "alzheimers_disease_data.csv"
KEY_COLUMNS = [
    "Age",
    "EducationLevel",
    "BMI",
    "MMSE",
    "FunctionalAssessment",
    "ADL",
    "FamilyHistoryAlzheimers",
]


def main():
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print("\nDiagnosis distribution:")
    print(df["Diagnosis"].value_counts().sort_index().to_string())
    print("Positive rate:", round(df["Diagnosis"].mean(), 4))
    print("\nFamilyHistoryAlzheimers distribution:")
    print(df["FamilyHistoryAlzheimers"].value_counts().sort_index().to_string())
    print("Positive rate:", round(df["FamilyHistoryAlzheimers"].mean(), 4))
    print("\nKey column min/max:")
    for name in KEY_COLUMNS:
        print(f"{name}: {df[name].min()} to {df[name].max()}")
    print("\nUnique values per column:")
    print(df.nunique().to_string())


if __name__ == "__main__":
    main()
