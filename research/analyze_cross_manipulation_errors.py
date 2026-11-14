
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = ROOT / "experiments" / "cross_manipulation_generalization"
OUTPUT_ROOT = INPUT_ROOT / "diagnostics"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "face2face",
    "faceswap",
    "neuraltextures"
]

MODELS = {
    "EfficientNet-B0": (
        "EfficientNet-B0_probability",
        "EfficientNet-B0_prediction"
    ),
    "DINOv2": (
        "DINOv2_probability",
        "DINOv2_prediction"
    ),
    "Fusion": (
        "B0_DINOv2_Fusion_probability",
        "B0_DINOv2_Fusion_prediction"
    )
}

summary = []
all_errors = []
all_predictions = []

for category in CATEGORIES:

    file_path = INPUT_ROOT / f"{category}_predictions.csv"
    df = pd.read_csv(file_path)

    df["category"] = category

    print("\n" + "=" * 65)
    print("CATEGORY:", category.upper())
    print("=" * 65)

    for model, columns in MODELS.items():

        probability_column, prediction_column = columns

        actual = df["actual_label"].to_numpy()
        predicted = df[prediction_column].to_numpy()
        probability = df[probability_column].to_numpy()

        correct = actual == predicted
        confidence = np.maximum(probability, 1 - probability)

        real_probability = probability[actual == 0]
        fake_probability = probability[actual == 1]

        wrong_confidence = confidence[~correct]

        high_confidence_wrong = (
            (~correct) & (confidence >= 0.90)
        )

        result = {
            "category": category,
            "model": model,
            "videos": len(df),
            "correct": int(correct.sum()),
            "incorrect": int((~correct).sum()),
            "accuracy": round(correct.mean(), 4),
            "mean_real_fake_probability": round(
                real_probability.mean(), 4
            ),
            "mean_fake_fake_probability": round(
                fake_probability.mean(), 4
            ),
            "probability_separation": round(
                fake_probability.mean() - real_probability.mean(), 4
            ),
            "mean_wrong_confidence": round(
                wrong_confidence.mean(), 4
            ) if len(wrong_confidence) else np.nan,
            "high_confidence_errors": int(
                high_confidence_wrong.sum()
            )
        }

        summary.append(result)

        print("\nMODEL:", model)
        print("Correct:", result["correct"])
        print("Incorrect:", result["incorrect"])
        print("Accuracy:", result["accuracy"])
        print("Mean fake probability for REAL:", result["mean_real_fake_probability"])
        print("Mean fake probability for FAKE:", result["mean_fake_fake_probability"])
        print("Probability separation:", result["probability_separation"])
        print("Mean confidence on wrong predictions:", result["mean_wrong_confidence"])
        print("Wrong predictions with confidence >= 0.90:", result["high_confidence_errors"])

        errors = df.loc[~correct, [
            "video_name",
            "actual_label",
            probability_column,
            prediction_column
        ]].copy()

        errors["category"] = category
        errors["model"] = model
        errors["confidence"] = confidence[~correct]

        all_errors.append(errors)

    b0_predictions = df["EfficientNet-B0_prediction"].to_numpy()
    dino_predictions = df["DINOv2_prediction"].to_numpy()

    disagreement = b0_predictions != dino_predictions

    print("\nB0-DINOv2 DISAGREEMENTS:", int(disagreement.sum()))

    disagreements = df.loc[disagreement].copy()
    disagreements["category"] = category

    all_predictions.append(df)


summary_df = pd.DataFrame(summary)
errors_df = pd.concat(all_errors, ignore_index=True)
predictions_df = pd.concat(all_predictions, ignore_index=True)

summary_path = OUTPUT_ROOT / "confidence_summary.csv"
errors_path = OUTPUT_ROOT / "misclassified_videos.csv"
predictions_path = OUTPUT_ROOT / "all_predictions.csv"

summary_df.to_csv(summary_path, index=False)
errors_df.to_csv(errors_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

print("\n" + "=" * 65)
print("FINAL CONFIDENCE ANALYSIS")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nMISCLASSIFIED VIDEO COUNT:", len(errors_df))

print("\nSAVED FILES:")
print(summary_path)
print(errors_path)
print(predictions_path)

print("\nCONFIDENCE AND ERROR ANALYSIS COMPLETED")
