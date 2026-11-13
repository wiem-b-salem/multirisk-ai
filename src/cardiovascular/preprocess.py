from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "cardiovascular" / "CHDdata.csv"
SEED = 42


def load_and_split(test_size=0.2):
    df = pd.read_csv(DATA_PATH)
    df["famhist"] = df["famhist"].map({"Absent": 0, "Present": 1})
    assert df["famhist"].notna().all()

    X = df.drop(columns="chd")
    y = df["chd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = load_and_split()
    print("Train:", X_train.shape, "positive rate:", round(y_train.mean(), 3))
    print("Test: ", X_test.shape, "positive rate:", round(y_test.mean(), 3))
    print("Train mean (should be ~0):", X_train.mean().abs().max().round(6))