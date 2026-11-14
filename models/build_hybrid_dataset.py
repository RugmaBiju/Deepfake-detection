
import numpy as np
import pandas as pd
from pathlib import Path

embedding_path = "datasets/metadata/efficientnet_temporal_features.npz"
forensic_path = "datasets/metadata/ffpp_hybrid_video_features.csv"

output_path = "datasets/metadata/ffpp_hybrid_dataset.csv"

data = np.load(embedding_path)

sequences = data["sequences"]
labels = data["labels"]
video_names = data["video_names"]
splits = data["splits"]

# Convert 10 frame embeddings into one video embedding
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

# Load forensic features
forensic_df = pd.read_csv(forensic_path)

# Keep only videos available in the EfficientNet dataset
forensic_df = forensic_df[
    forensic_df["video_name"].isin(video_names)
].copy()

# Merge using video names
hybrid_df = visual_df.merge(
    forensic_df,
    on="video_name",
    suffixes=("_visual", "_forensic"),
    validate="one_to_one"
)

# Verify labels and splits match
if not np.array_equal(
    hybrid_df["label_visual"].to_numpy(),
    hybrid_df["label_forensic"].to_numpy()
):
    raise ValueError("Label mismatch detected")

if not np.array_equal(
    hybrid_df["split_visual"].to_numpy(),
    hybrid_df["split_forensic"].to_numpy()
):
    raise ValueError("Split mismatch detected")

hybrid_df["label"] = hybrid_df["label_visual"]
hybrid_df["split"] = hybrid_df["split_visual"]

hybrid_df = hybrid_df.drop(
    columns=["label_visual", "label_forensic",
             "split_visual", "split_forensic",
             "group_id_visual", "group_id_forensic",
             "category_visual", "category_forensic"],
    errors="ignore"
)

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
hybrid_df.to_csv(output_path, index=False)

print("\nHYBRID DATASET CREATED")
print("Videos:", len(hybrid_df))
print("Visual features:", len(visual_columns))
print("Forensic features:", 14)
print("Total features:", len(visual_columns) + 14)
print("\nSplit distribution:")
print(hybrid_df["split"].value_counts())
print("\nLabel distribution:")
print(hybrid_df["label"].value_counts())
print("\nSaved to:", output_path)
