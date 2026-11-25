from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd

from src.alzheimers.preprocess import DATA_PATH, load_and_split as load_alzheimers
from src.family_history_check import logreg_coef, report_association

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "results" / "alzheimers_family_history.txt"


def main():
    df = pd.read_csv(DATA_PATH)
    buffer = StringIO()
    with redirect_stdout(buffer):
        report_association(
            "Alzheimer's (FamilyHistoryAlzheimers)",
            df["FamilyHistoryAlzheimers"],
            df["Diagnosis"],
        )
        print("=== Logistic regression coefficients (standardized) ===")
        for setup in ("full", "risk_only", "compact"):
            loader = lambda selected=setup: load_alzheimers(selected)
            print(f"{setup}: {logreg_coef(loader, 'FamilyHistoryAlzheimers'):.4f}")
    output = buffer.getvalue()
    print(output, end="")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output)


if __name__ == "__main__":
    main()
