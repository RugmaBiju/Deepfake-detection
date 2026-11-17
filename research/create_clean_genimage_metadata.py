
import hashlib
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("datasets/metadata/genimage_metadata.csv")

df["split_priority"] = df["split"].map({
    "train": 0,
    "validation": 1,
    "test": 2
})

hashes = []

print("Hashing dataset images...")

for path in tqdm(df["image_path"]):
    with open(path, "rb") as file:
        hashes.append(hashlib.sha256(file.read()).hexdigest())

df["image_hash"] = hashes

df = df.sort_values("split_priority")

clean_df = df.drop_duplicates(
    subset="image_hash",
    keep="first"
).copy()

clean_df = clean_df.drop(
    columns=["split_priority", "image_hash"]
)

output_path = "datasets/metadata/genimage_metadata_deduplicated.csv"

clean_df.to_csv(output_path, index=False)

print("\nCLEANED DATASET")
print(pd.crosstab(clean_df["split"], clean_df["category"]))
print("\nOriginal rows:", len(hashes))
print("Cleaned rows:", len(clean_df))
print("Removed duplicates:", len(hashes) - len(clean_df))
print("Saved:", output_path)
