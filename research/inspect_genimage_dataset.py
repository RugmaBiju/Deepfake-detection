
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

metadata_path = "datasets/metadata/genimage_metadata.csv"
output_path = "datasets/metadata/genimage_image_inspection.csv"

df = pd.read_csv(metadata_path)

results = []

print("Inspecting", len(df), "images...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    image_path = row["image_path"]

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            image_format = img.format

        results.append({
            "filename": row["filename"],
            "category": row["category"],
            "split": row["split"],
            "width": width,
            "height": height,
            "mode": mode,
            "format": image_format,
            "status": "valid"
        })

    except Exception as e:
        results.append({
            "filename": row["filename"],
            "category": row["category"],
            "split": row["split"],
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "status": "corrupted"
        })

inspection = pd.DataFrame(results)

inspection.to_csv(output_path, index=False)

print("\nINSPECTION SUMMARY")
print(inspection["status"].value_counts())
print("\nIMAGE DIMENSIONS")
print(inspection.groupby(["category", "width", "height"]).size().head(20))
print("\nIMAGE MODES")
print(inspection["mode"].value_counts())
print("\nIMAGE FORMATS")
print(inspection["format"].value_counts())

print("\nInspection saved to:", output_path)
