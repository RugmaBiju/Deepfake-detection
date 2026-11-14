
import os
import pandas as pd
from sklearn.model_selection import train_test_split

base_path = "datasets/processed/genimage"
output_path = "datasets/metadata"

os.makedirs(output_path, exist_ok=True)

categories = ["Real", "DDPM", "DDIM"]

real_files = sorted([
    f for f in os.listdir(os.path.join(base_path, "Real"))
    if f.lower().endswith(".png")
])

ddpm_files = set(os.listdir(os.path.join(base_path, "DDPM")))
ddim_files = set(os.listdir(os.path.join(base_path, "DDIM")))

if set(real_files) != ddpm_files or set(real_files) != ddim_files:
    raise ValueError("Filename mismatch between categories!")

print("Verified matching filename groups:", len(real_files))

train_groups, remaining_groups = train_test_split(
    real_files,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

validation_groups, test_groups = train_test_split(
    remaining_groups,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

split_map = {}

for filename in train_groups:
    split_map[filename] = "train"

for filename in validation_groups:
    split_map[filename] = "validation"

for filename in test_groups:
    split_map[filename] = "test"

rows = []

for filename in real_files:
    for category in categories:

        if category == "Real":
            label = 0
        else:
            label = 1

        rows.append({
            "filename": filename,
            "group_id": filename,
            "category": category,
            "label": label,
            "split": split_map[filename],
            "image_path": os.path.join(base_path, category, filename)
        })

df = pd.DataFrame(rows)

df.to_csv(
    os.path.join(output_path, "genimage_metadata.csv"),
    index=False
)

print("\nDATASET SPLIT SUMMARY")
print(df.groupby(["split", "category"]).size())

print("\nTOTAL IMAGES:", len(df))
print("TRAIN GROUPS:", len(train_groups))
print("VALIDATION GROUPS:", len(validation_groups))
print("TEST GROUPS:", len(test_groups))

print("\nMetadata saved successfully.")
