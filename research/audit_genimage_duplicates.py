
import hashlib
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

df = pd.read_csv("datasets/metadata/genimage_metadata.csv")

hash_records = defaultdict(list)

for _, row in tqdm(df.iterrows(), total=len(df)):

    with open(row["image_path"], "rb") as file:
        image_hash = hashlib.sha256(file.read()).hexdigest()

    hash_records[image_hash].append({
        "filename": row["filename"],
        "category": row["category"],
        "split": row["split"],
        "image_path": row["image_path"]
    })

print("\nCROSS-SPLIT DUPLICATE DETAILS")

for image_hash, records in hash_records.items():

    splits = set(record["split"] for record in records)

    if len(splits) > 1:

        print("\nHash:", image_hash)

        for record in records:
            print(
                "Split:", record["split"],
                "| Category:", record["category"],
                "| Filename:", record["filename"],
                "| Path:", record["image_path"]
            )

print("\nAUDIT COMPLETE")
