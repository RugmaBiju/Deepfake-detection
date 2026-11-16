import os
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "prnu_block_mean",
    "prnu_block_std",
    "prnu_block_min",

    "pg_a",
    "pg_b",
    "pg_fit_rmse",
    "pg_num_flat_patches",

    "hf_energy_ratio",
    "hf_residual_energy",
    "hf_residual_std",
    "hf_residual_mean_abs",
    "hf_residual_max_abs",
]


FEATURE_FILE = "data/features_v2.csv"
MODEL_DIR = "models_saved"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "baseline_xgb_v2.pkl",
)


def main():

    print("=" * 70)
    print("TRUTHLENS — XGBOOST TRAINING")
    print("=" * 70)

    # --------------------------------------------------
    # Load features
    # --------------------------------------------------

    print("\nLoading:", FEATURE_FILE)

    df = pd.read_csv(FEATURE_FILE)

    print("Total usable samples:", len(df))

    # --------------------------------------------------
    # Check features
    # --------------------------------------------------

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing features: {missing}"
        )

    # --------------------------------------------------
    # Convert features to numeric
    # --------------------------------------------------

    for feature in FEATURE_COLUMNS:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    # Remove rows with invalid feature values

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["label"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # X and y
    # --------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )

    y = (
        df["label"]
        .astype(str)
        .str.lower()
        .eq("fake")
        .astype(int)
        .to_numpy()
    )

    print("\nReal samples:", np.sum(y == 0))
    print("Fake samples:", np.sum(y == 1))

    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------

    print("\nTraining XGBoost...")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"AUC:      {auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "real",
                "fake",
            ],
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print("Confusion Matrix:")

    print(cm)

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importances = model.feature_importances_

    ranking = sorted(
        zip(
            FEATURE_COLUMNS,
            importances,
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    for feature, importance in ranking:

        print(
            f"{feature:30s} "
            f"{importance:.4f}"
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    with open(
        MODEL_FILE,
        "wb",
    ) as f:

        pickle.dump(
            model,
            f,
        )

    print(
        f"\nModel saved to: {MODEL_FILE}"
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()