
import numpy as np
import pandas as pd

from pathlib import Path

visual_path = Path("datasets/metadata/efficientnet_temporal_features.npz")
forensic_path = Path("datasets/metadata/ffpp_face_forensic_video_features.csv")

output_path = Path("datasets/metadata/ffpp_face_hybrid_dataset.csv")

# Load EfficientNet temporal embeddings
data = np.load(visual_path)

sequences = data["sequences"]
labels = data["labels"]
video_names = data["video_names"]
splits = data["splits"]

# Average frame embeddings to obtain video-level visual features
visual_features = sequences.mean(axis=1)

visual_columns = [
    f"visual_{i}" for i in range(visual_features.shape[1])
]

visual_df = pd.DataFrame(
    visual_features,
    columns=visual_columns
)

visual_df["video_name"] = video_names
visual_df["label"] = labels
visual_df["split"] = splits

# Load face-focused forensic features
forensic_df = pd.read_csv(forensic_path)

if forensic_df["video_name"].duplicated().any():
    raise ValueError("Duplicate video names in forensic dataset")

# Verify labels and splits for matching videos
merged = visual_df.merge(
    forensic_df,
    on="video_name",
    how="inner",
    suffixes=("_visual", "_forensic"),
    validate="one_to_one"
)

if not (merged["label_visual"] == merged["label_forensic"]).all():
    raise ValueError("Label mismatch detected")

if not (merged["split_visual"] == merged["split_forensic"]).all():
    raise ValueError("Split mismatch detected")

# Keep the shared label and split
merged["label"] = merged["label_visual"]
merged["split"] = merged["split_visual"]

merged = merged.drop(
    columns=[
        "label_visual",
        "label_forensic",
        "split_visual",
        "split_forensic",
        "group_id",
        "category"
    ],
    errors="ignore"
)

# Save dataset
merged.to_csv(output_path, index=False)

forensic_columns = [
    column for column in forensic_df.columns
    if column not in [
        "video_name", "group_id", "split", "label", "category"
    ]
]

print("\nFACE-FOCUSED HYBRID DATASET CREATED")
print("Videos:", len(merged))
print("Visual features:", len(visual_columns))
print("Face forensic features:", len(forensic_columns))
print("Total features:", len(visual_columns) + len(forensic_columns))

print("\nSplit distribution:")
print(merged["split"].value_counts())

print("\nLabel distribution:")
print(merged["label"].value_counts())

print("\nSplit and label distribution:")
print(merged.groupby(["split", "label"]).size())

print("\nSaved to:", output_path)
