
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

data = np.load(
    "datasets/metadata/convnext_features/convnext_trainval_features_clean.npz"
)

X = data["features"]
y = data["label"]
categories = np.char.lower(data["category"].astype(str))
splits = np.char.lower(data["split"].astype(str))

test = np.load(
    "datasets/metadata/convnext_features/convnext_test_features.npz"
)

X_test_all = test["features"]
y_test_all = test["labels"]
categories_test = np.char.lower(test["categories"].astype(str))

experiments = {
    "A": {
        "seen": ["face2face", "faceswap"],
        "unseen": "neuraltextures"
    },
    "B": {
        "seen": ["face2face", "neuraltextures"],
        "unseen": "faceswap"
    },
    "C": {
        "seen": ["faceswap", "neuraltextures"],
        "unseen": "face2face"
    }
}

C_values = [0.001, 0.01, 0.1, 1, 10, 100]

results = {}

for name, config in experiments.items():

    print("\n" + "=" * 60)
    print("CONVNEXT LOMO TEST EXPERIMENT", name)
    print("Seen:", config["seen"])
    print("Unseen:", config["unseen"])
    print("=" * 60)

    train_mask = (
        (splits == "train") &
        (
            (categories == "real") |
            np.isin(categories, config["seen"])
        )
    )

    val_mask = (
        (splits == "validation") &
        (
            (categories == "real") |
            np.isin(categories, config["seen"])
        )
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_c = None
    best_f1 = -1

    for c in C_values:

        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        val_predictions = model.predict(X_val_scaled)

        score = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        print(f"C: {c} | Validation F1: {score:.4f}")

        if score > best_f1:
            best_f1 = score
            best_c = c

    final_model = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    )

    final_model.fit(X_train_scaled, y_train)

    test_mask = (
        (categories_test == "real") |
        (categories_test == config["unseen"])
    )

    X_test = X_test_all[test_mask]
    y_test = y_test_all[test_mask]

    X_test_scaled = scaler.transform(X_test)

    predictions = final_model.predict(X_test_scaled)
    probabilities = final_model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "best_C": best_c,
        "validation_f1": float(best_f1),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
    }

    results[name] = metrics

    print("\nSelected C:", best_c)
    print("Test metrics:", metrics)

print("\n" + "=" * 60)
print("CONVNEXT LOMO TEST SUMMARY")
print("=" * 60)

for name, metrics in results.items():
    print(
        name,
        "| Accuracy:", round(metrics["test_accuracy"], 4),
        "| F1:", round(metrics["test_f1"], 4),
        "| AUC:", round(metrics["test_roc_auc"], 4)
    )

with open(
    "experiments/convnext_lomo_test_metrics.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\nSaved: experiments/convnext_lomo_test_metrics.json")
