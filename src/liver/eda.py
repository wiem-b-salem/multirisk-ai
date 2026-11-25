from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "liver" / "Liver_disease_data.csv"
KEY_COLUMNS = [
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
    print("\nGeneticRisk distribution:")
    print(df["GeneticRisk"].value_counts().sort_index().to_string())
    print("Positive rate by level:")
    print(df.groupby("GeneticRisk")["Diagnosis"].mean().round(4).to_string())
    print("\nKey column min/max:")
    for name in KEY_COLUMNS:
        print(f"{name}: {df[name].min()} to {df[name].max()}")
    print("\nUnique values per column:")
    print(df.nunique().to_string())


if __name__ == "__main__":
    main()
