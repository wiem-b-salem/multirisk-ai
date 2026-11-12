import pandas as pd
import numpy as np

# Load both datasets
df_diabetes = pd.read_csv('data/diabetes/diabetes_data.csv')
df_cvd = pd.read_csv('data/cardiovascular/CHDdata.csv')

print("=" * 60)
print("DIABETES DATASET")
print("=" * 60)
print(f"Shape: {df_diabetes.shape}")
print(f"\nMissing values:\n{df_diabetes.isnull().sum()}")
print(f"\nFamily History distribution:\n{df_diabetes['FamilyHistoryDiabetes'].value_counts()}")
print(f"\nOutcome (Diagnosis) distribution:\n{df_diabetes['Diagnosis'].value_counts()}")
print(f"\nBasic stats:\n{df_diabetes.describe()}")

print("\n" + "=" * 60)
print("CVD DATASET")
print("=" * 60)
print(f"Shape: {df_cvd.shape}")
print(f"\nMissing values:\n{df_cvd.isnull().sum()}")
print(f"\nFamily History distribution:\n{df_cvd['famhist'].value_counts()}")
print(f"\nOutcome (CHD) distribution:\n{df_cvd['chd'].value_counts()}")
print(f"\nBasic stats:\n{df_cvd.describe()}")