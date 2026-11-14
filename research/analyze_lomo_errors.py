
import os
import pandas as pd

BASE_DIR = "experiments"
OUTPUT_DIR = os.path.join(BASE_DIR, "lomo_error_analysis")

MODELS = {
    "B0": os.path.join(BASE_DIR, "lomo_generalization"),
    "DINOv2": os.path.join(BASE_DIR, "lomo_generalization_dinov2"),
    "Fusion": os.path.join(BASE_DIR, "lomo_generalization_fusion")
}

EXPERIMENTS = ["A", "B", "C"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_predictions = []
summary_rows = []
error_rows = []

for experiment in EXPERIMENTS:
    model_predictions = []

    for model_name, folder in MODELS.items():
        file_path = os.path.join(
            folder,
            f"experiment_{experiment}_test_predictions.csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

        df = pd.read_csv(file_path)

        required_columns = {
            "video_name",
            "actual_label",
            "predicted_label",
            "fake_probability"
        }

        if not required_columns.issubset(df.columns):
            raise ValueError(f"Unexpected columns in {file_path}")

        df = df[
            [
                "video_name",
                "actual_label",
                "predicted_label",
                "fake_probability"
            ]
        ].copy()

        df = df.rename(columns={
            "predicted_label": f"{model_name}_prediction",
            "fake_probability": f"{model_name}_fake_probability"
        })

        model_predictions.append(df)

        actual = df["actual_label"]
        predicted = df[f"{model_name}_prediction"]

        tn = int(((actual == 0) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        tp = int(((actual == 1) & (predicted == 1)).sum())

        summary_rows.append({
            "experiment": experiment,
            "unseen_manipulation": {
                "A": "NeuralTextures",
                "B": "FaceSwap",
                "C": "Face2Face"
            }[experiment],
            "model": model_name,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "false_positives": fp,
            "false_negatives": fn
        })

        errors = df[actual != predicted].copy()
        errors["experiment"] = experiment
        errors["model"] = model_name

        errors["error_type"] = errors.apply(
            lambda row: (
                "False Positive"
                if row["actual_label"] == 0
                else "False Negative"
            ),
            axis=1
        )

        error_rows.append(errors)

    # Merge predictions from all three models by video name.
    merged = model_predictions[0]

    for next_df in model_predictions[1:]:
        merged = merged.merge(
            next_df,
            on=["video_name", "actual_label"],
            how="inner",
            validate="one_to_one"
        )

    if len(merged) != 32:
        raise ValueError(
            f"Experiment {experiment}: expected 32 aligned videos, "
            f"found {len(merged)}"
        )

    prediction_columns = [
        "B0_prediction",
        "DINOv2_prediction",
        "Fusion_prediction"
    ]

    merged["models_correct"] = sum(
        merged[column] == merged["actual_label"]
        for column in prediction_columns
    )

    merged["experiment"] = experiment

    all_predictions.append(merged)

    print(f"\nExperiment {experiment}")
    print(f"Aligned test videos: {len(merged)}")

summary = pd.DataFrame(summary_rows)
errors = pd.concat(error_rows, ignore_index=True)
comparison = pd.concat(all_predictions, ignore_index=True)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix_summary.csv"),
    index=False
)

errors.to_csv(
    os.path.join(OUTPUT_DIR, "misclassified_videos.csv"),
    index=False
)

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "model_prediction_comparison.csv"),
    index=False
)

print("\n" + "=" * 60)
print("LOMO ERROR ANALYSIS SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))

print("\n" + "=" * 60)
print("ERROR COUNTS")
print("=" * 60)

print(
    errors.groupby(
        ["experiment", "model", "error_type"]
    ).size().unstack(fill_value=0).to_string()
)

print("\n" + "=" * 60)
print("MODEL AGREEMENT")
print("=" * 60)

print(
    comparison.groupby("experiment")["models_correct"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nFiles saved in:", OUTPUT_DIR)
