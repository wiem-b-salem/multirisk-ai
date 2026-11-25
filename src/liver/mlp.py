import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.liver.preprocess import load_and_split as load_liver

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "results" / "liver_mlp.csv"
MODELS_DIR = ROOT / "models" / "liver"

SETUPS = {
    "liver_full": lambda: load_liver("full"),
    "liver_no_lab": lambda: load_liver("no_lab"),
}


def evaluate(y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "test_roc_auc": roc_auc_score(y_true, y_proba),
        "test_pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def build_model(n_features, seed):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


def fit_model(model, X_train, y_train, seed, val_size):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )
    classes = np.unique(y_fit)
    weights = compute_class_weight("balanced", classes=classes, y=y_fit)
    class_weight = dict(zip(classes, weights))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0,
    )
    return model


def main():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for setup, loader in SETUPS.items():
        X_train, X_test, y_train, y_test, scaler = loader()
        n_features = X_train.shape[1]

        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model = build_model(n_features, SEED)
            fit_model(
                model,
                X_train.iloc[train_idx],
                y_train.iloc[train_idx],
                SEED,
                val_size=0.15,
            )
            proba = model.predict(X_train.iloc[val_idx], verbose=0).ravel()
            scores.append(roc_auc_score(y_train.iloc[val_idx], proba))

        model = build_model(n_features, SEED)
        fit_model(model, X_train, y_train, SEED, val_size=0.2)
        y_proba = model.predict(X_test, verbose=0).ravel()
        metrics = evaluate(y_test, y_proba)

        seed_aucs = []
        for seed in range(5):
            seed_model = build_model(n_features, seed)
            fit_model(seed_model, X_train, y_train, seed, val_size=0.2)
            seed_proba = seed_model.predict(X_test, verbose=0).ravel()
            seed_aucs.append(roc_auc_score(y_test, seed_proba))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODELS_DIR / f"mlp_{setup}.keras")
        joblib.dump(scaler, MODELS_DIR / f"scaler_{setup}.joblib")

        rows.append(
            {
                "setup": setup,
                "model": "mlp",
                "cv_roc_auc_mean": np.mean(scores),
                "cv_roc_auc_std": np.std(scores),
                **metrics,
                "test_roc_auc_seed_mean": np.mean(seed_aucs),
                "test_roc_auc_seed_std": np.std(seed_aucs),
            }
        )

    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
