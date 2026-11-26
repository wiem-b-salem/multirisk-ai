from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.baselines import MODELS
from src.liver.preprocess import DATA_PATH, load_and_split as load_liver

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "results" / "liver_genetic_risk.txt"
SETUPS = ("full", "no_lab")


def logreg_coef(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    model.fit(X_train, y_train)
    return model.coef_[0][list(X_train.columns).index("GeneticRisk")]


def odds_ratio_and_p(pos_base, neg_base, pos_level, neg_level):
    odds_ratio = (pos_level / neg_level) / (pos_base / neg_base)
    p_value = chi2_contingency([[pos_base, neg_base], [pos_level, neg_level]])[1]
    return odds_ratio, p_value


def main():
    df = pd.read_csv(DATA_PATH)
    counts = df.groupby("GeneticRisk")["Diagnosis"].agg(["count", "sum", "mean"])
    table = [
        [int(counts.loc[level, "sum"]), int(counts.loc[level, "count"] - counts.loc[level, "sum"])]
        for level in (0, 1, 2)
    ]
    chi2, p_value, dof, _ = chi2_contingency(table)

    buffer = StringIO()
    with redirect_stdout(buffer):
        print("=== Liver disease (GeneticRisk) ===")
        print("Positive rate by genetic risk level:")
        for level in (0, 1, 2):
            positive = table[level][0]
            negative = table[level][1]
            print(
                f"Level {level}: n={positive + negative}, positive={positive}, "
                f"rate={counts.loc[level, 'mean']:.4f}"
            )
        print(
            f"Chi-square across levels 0/1/2: chi2={chi2:.4f}, dof={dof}, "
            f"p-value={p_value:.6g}"
        )
        for level in (1, 2):
            odds_ratio, level_p = odds_ratio_and_p(*table[0], *table[level])
            print(
                f"Level {level} vs 0: odds ratio={odds_ratio:.4f}, "
                f"chi-square p-value={level_p:.6g}"
            )

        print("\n=== Logistic regression coefficients (standardized) ===")
        for setup in SETUPS:
            X_train, _, y_train, _, _ = load_liver(setup)
            print(f"{setup}: {logreg_coef(X_train, y_train):.4f}")

        print("\n=== Added value of GeneticRisk (training CV ROC-AUC) ===")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for setup in SETUPS:
            X_train, _, y_train, _, _ = load_liver(setup)
            variants = {
                "with": X_train,
                "without": X_train.drop(columns="GeneticRisk"),
            }
            for variant, X in variants.items():
                for name, factory in MODELS.items():
                    scores = cross_val_score(factory(), X, y_train, cv=cv, scoring="roc_auc")
                    print(
                        f"{setup} {variant} {name}: "
                        f"{scores.mean():.4f} +/- {scores.std():.4f}"
                    )

    output = buffer.getvalue()
    print(output, end="")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output)


if __name__ == "__main__":
    main()
