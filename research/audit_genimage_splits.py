
import pandas as pd

df = pd.read_csv("datasets/metadata/genimage_metadata.csv")

print("\nDATASET SPLIT COUNTS")
print(pd.crosstab(df["split"], df["category"]))

print("\nGROUP ID OVERLAP CHECK")

train_groups = set(df[df["split"] == "train"]["group_id"])
val_groups = set(df[df["split"] == "validation"]["group_id"])
test_groups = set(df[df["split"] == "test"]["group_id"])

print("Train groups:", len(train_groups))
print("Validation groups:", len(val_groups))
print("Test groups:", len(test_groups))

print("Train-Val overlap:", len(train_groups & val_groups))
print("Train-Test overlap:", len(train_groups & test_groups))
print("Val-Test overlap:", len(val_groups & test_groups))

print("\nIMAGE PATH CHECK")
print("Total rows:", len(df))
print("Unique image paths:", df["image_path"].nunique())
print("Duplicate paths:", df["image_path"].duplicated().sum())
print("Missing paths:", sum(
    not __import__("os").path.exists(path)
    for path in df["image_path"]
))

print("\nAUDIT COMPLETE")
