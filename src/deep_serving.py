import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.mlp import SEED, build_model, evaluate, fit_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "dl_risk_thresholds.json"
METRICS_CONFIG_PATH = ROOT / "config" / "comparison_metrics.json"
RESULTS_PATH = ROOT / "results" / "comparison_metrics.csv"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

SERVED = {
    "cvd": {
        "folder": "cardiovascular",
        "setup": "cvd",
        "csv": ROOT / "data" / "cardiovascular" / "CHDdata.csv",
        "target": "chd",
        "dl_file": "mlp_cvd.keras",
    },
    "diabetes": {
        "folder": "diabetes",
        "setup": "diabetes_compact",
        "csv": ROOT / "data" / "diabetes" / "diabetes_data.csv",
        "target": "Diagnosis",
        "dl_file": "mlp_diabetes_compact.keras",
    },
    "alzheimers": {
        "folder": "alzheimers",
        "setup": "alzheimers_compact",
        "csv": ROOT / "data" / "alzheimers" / "alzheimers_disease_data.csv",
        "target": "Diagnosis",
        "dl_file": "mlp_alzheimers_compact.keras",
    },
    "liver": {
        "folder": "liver",
        "setup": "liver_full",
        "csv": ROOT / "data" / "liver" / "Liver_disease_data.csv",
        "target": "Diagnosis",
        "dl_file": "mlp_liver_full.keras",
    },
}


def schema_for(config):
    path = ROOT / "models" / config["folder"] / f"schema_{config['setup']}.json"
    return json.loads(path.read_text())


def raw_split(config, schema):
    frame = pd.read_csv(config["csv"])
    if config["disease"] == "cvd":
        frame["famhist"] = frame["famhist"].map({"Absent": 0, "Present": 1})
    X = frame[schema["feature_columns"]]
    y = frame[config["target"]]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def scaled_frames(scaler, X_train, X_test, columns):
    train = pd.DataFrame(scaler.transform(X_train), columns=columns, index=X_train.index)
    test = pd.DataFrame(scaler.transform(X_test), columns=columns, index=X_test.index)
    return train, test


def choose_thresholds(scores, y):
    candidates = np.unique(scores)
    recalls = np.array([
        np.sum((scores >= value) & (y == 1)) / np.sum(y == 1)
        for value in candidates
    ])
    specificities = np.array([
        np.sum((scores < value) & (y == 0)) / np.sum(y == 0)
        for value in candidates
    ])
    eligible_low = candidates[recalls >= 0.90]
    eligible_high = candidates[specificities >= 0.90]
    low_max = float(eligible_low.max()) if len(eligible_low) else float(candidates.min())
    high_min = float(eligible_high.min()) if len(eligible_high) else float(candidates.max())
    if low_max >= high_min:
        threshold = float(candidates[np.argmax(recalls + specificities - 1)])
        low_max = high_min = threshold
    return {"low_max": low_max, "high_min": high_min}


def main():
    threshold_config = {}
    metrics_config = {}
    rows = []
    for disease, config in SERVED.items():
        config["disease"] = disease
        schema = schema_for(config)
        X_train_raw, X_test_raw, y_train, y_test = raw_split(config, schema)
        columns = schema["feature_columns"]
        n_features = len(columns)
        y_train_array = y_train.to_numpy()

        final_scaler = StandardScaler().fit(X_train_raw)
        X_train, X_test = scaled_frames(final_scaler, X_train_raw, X_test_raw, columns)

        started = time.perf_counter()
        model = build_model(n_features, SEED)
        fit_model(model, X_train, y_train, SEED, val_size=0.2)
        dl_training_seconds = time.perf_counter() - started

        out_dir = ROOT / "models" / config["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / config["dl_file"])
        joblib.dump(final_scaler, out_dir / schema["scaler_file"])

        dl_scores = model.predict(X_test, verbose=0).ravel()
        dl_metrics = evaluate(y_test.to_numpy(), dl_scores)
        dl_metrics["test_accuracy"] = accuracy_score(
            y_test.to_numpy(), (dl_scores >= 0.5).astype(int)
        )

        oof = np.zeros(len(y_train_array))
        for train_idx, val_idx in CV.split(X_train_raw, y_train_array):
            fold_scaler = StandardScaler().fit(X_train_raw.iloc[train_idx])
            fold_train, fold_val = scaled_frames(
                fold_scaler,
                X_train_raw.iloc[train_idx],
                X_train_raw.iloc[val_idx],
                columns,
            )
            fold_model = build_model(n_features, SEED)
            fit_model(
                fold_model,
                fold_train,
                y_train_array[train_idx],
                SEED,
                val_size=0.15,
            )
            oof[val_idx] = fold_model.predict(fold_val, verbose=0).ravel()

        threshold_config[disease] = choose_thresholds(oof, y_train_array)

        ml_model = joblib.load(out_dir / schema["model_file"])
        ml_scores = ml_model.predict_proba(X_test)[:, 1]
        ml_metrics = evaluate(y_test.to_numpy(), ml_scores)
        ml_metrics["test_accuracy"] = accuracy_score(
            y_test.to_numpy(), (ml_scores >= 0.5).astype(int)
        )
        metrics_config[disease] = {
            "ml": {
                "model_file": schema["model_file"],
                "test_roc_auc": ml_metrics["test_roc_auc"],
                "test_accuracy": ml_metrics["test_accuracy"],
                "training_seconds": None,
            },
            "dl": {
                "model_file": config["dl_file"],
                "cv_roc_auc_mean": roc_auc_score(y_train_array, oof),
                "test_roc_auc": dl_metrics["test_roc_auc"],
                "test_accuracy": dl_metrics["test_accuracy"],
                "training_seconds": dl_training_seconds,
            },
        }

        common = {
            "disease": disease,
            "setup": config["setup"],
            "features": n_features,
            "train_rows": len(y_train_array),
            "test_rows": len(y_test),
        }
        rows.append({
            **common,
            "model": "ml",
            "model_file": schema["model_file"],
            "cv_roc_auc_mean": "",
            "cv_roc_auc_std": "",
            **ml_metrics,
            "training_seconds": "",
        })
        rows.append({
            **common,
            "model": "dl",
            "model_file": config["dl_file"],
            "cv_roc_auc_mean": roc_auc_score(y_train_array, oof),
            "cv_roc_auc_std": "",
            **dl_metrics,
            "training_seconds": dl_training_seconds,
        })
        print(
            f"{disease}: ML ROC-AUC {ml_metrics['test_roc_auc']:.4f}, "
            f"DL ROC-AUC {dl_metrics['test_roc_auc']:.4f}, "
            f"DL train {dl_training_seconds:.1f}s"
        )

    CONFIG_PATH.write_text(json.dumps(threshold_config, indent=2))
    METRICS_CONFIG_PATH.write_text(json.dumps(metrics_config, indent=2))
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {CONFIG_PATH}")
    print(f"Wrote {METRICS_CONFIG_PATH}")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
