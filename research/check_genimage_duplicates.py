
import hashlib
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

metadata_path = "datasets/metadata/genimage_metadata.csv"

output_path = "datasets/metadata/genimage_exact_duplicates.csv"

df = pd.read_csv(metadata_path)

hash_groups = defaultdict(list)

print("Calculating SHA-256 hashes for", len(df), "images...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    image_path = row["image_path"]

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    image_hash = sha256.hexdigest()

    hash_groups[image_hash].append({
        "filename": row["filename"],
        "category": row["category"],
        "split": row["split"],
        "image_path": image_path
    })

duplicates = []

for image_hash, images in hash_groups.items():

    if len(images) > 1:

        for image in images:
            duplicates.append({
                "sha256": image_hash,
                "filename": image["filename"],
                "category": image["category"],
                "split": image["split"],
                "image_path": image["image_path"],
                "duplicate_group_size": len(images)
            })

duplicate_df = pd.DataFrame(duplicates)

duplicate_df.to_csv(output_path, index=False)

duplicate_groups = sum(
    1 for images in hash_groups.values() if len(images) > 1
)

duplicate_images = len(duplicate_df)

print("\nEXACT DUPLICATE REPORT")
print("Total images checked:", len(df))
print("Unique SHA-256 hashes:", len(hash_groups))
print("Duplicate groups:", duplicate_groups)
print("Images belonging to duplicate groups:", duplicate_images)
print("Extra duplicate copies:", duplicate_images - duplicate_groups)

if not duplicate_df.empty:
    print("\nDuplicates by category:")
    print(duplicate_df.groupby("category").size())

    print("\nDuplicates across splits:")
    print(duplicate_df.groupby("split").size())

print("\nReport saved to:", output_path)
