"""
The most important evaluation for this project: train on features from
one set of generators, test on a held-out generator the model has NEVER
seen. This tells you whether the PRNU/Poisson-Gaussian/HF features
capture something fundamental about diffusion generation vs. real
sensors, or whether the model just memorized artifacts specific to one
generator's architecture.

Run:
    python src/evaluation/cross_generator_eval.py \
        --features data/features.csv \
        --train_generators Real DDIM SDv15_DS0.5 \
        --test_generators LDM

Any generator not listed in --train_generators or --test_generators is
simply ignored for that run.
"""
import argparse

import pandas as pd
import xgboost as xgb

from src.models.train_baseline import FEATURE_COLUMNS
from src.evaluation.metrics import full_report


def run(features_csv: str, train_generators: list[str], test_generators: list[str]):
    df = pd.read_csv(features_csv).dropna(subset=FEATURE_COLUMNS)

    # "Real" should typically appear in BOTH train and test splits, since
    # every eval needs real examples to compare against. Only the FAKE
    # generators should differ between train/test.
    train_df = df[df["generator"].isin(train_generators)]
    test_df = df[df["generator"].isin(test_generators) | (df["generator"] == "Real")]

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = (train_df["label"] == "fake").astype(int).values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = (test_df["label"] == "fake").astype(int).values

    print(f"Train: {len(train_df)} rows from {sorted(train_df['generator'].unique())}")
    print(f"Test:  {len(test_df)} rows from {sorted(test_df['generator'].unique())}")

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = full_report(y_test, y_pred, y_prob)
    print("\n=== Cross-generator generalization report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/features.csv")
    parser.add_argument("--train_generators", nargs="+", required=True)
    parser.add_argument("--test_generators", nargs="+", required=True)
    args = parser.parse_args()

    run(args.features, args.train_generators, args.test_generators)
