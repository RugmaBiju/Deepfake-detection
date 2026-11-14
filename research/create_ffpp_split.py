
from pathlib import Path
from collections import defaultdict
import random
import json
import pandas as pd

real_path = Path(r"C:\FaceForensics++\original_sequences\youtube\c23\videos")
fake_path = Path(r"C:\FaceForensics++\manipulated_sequences\Deepfakes\c23\videos")

real_files = {p.stem: p for p in real_path.glob("*.mp4")}
fake_files = list(fake_path.glob("*.mp4"))

groups = defaultdict(list)

for file in fake_files:
    a, b = file.stem.split("_")
    group_id = tuple(sorted((a, b)))
    groups[group_id].append(file)

for group, files in groups.items():
    expected = {f"{group[0]}_{group[1]}", f"{group[1]}_{group[0]}"}

    if {f.stem for f in files} != expected:
        raise ValueError(f"Incomplete fake pair: {group}")

    if group[0] not in real_files or group[1] not in real_files:
        raise ValueError(f"Missing real video: {group}")

group_list = sorted(groups.keys())

if len(group_list) != 50:
    raise ValueError(f"Expected 50 groups, found {len(group_list)}")

random.Random(42).shuffle(group_list)

train_groups = group_list[:35]
val_groups = group_list[35:42]
test_groups = group_list[42:50]

split_groups = {
    "train": train_groups,
    "validation": val_groups,
    "test": test_groups
}

rows = []
split_data = {
    "seed": 42,
    "method": "identity_grouped",
    "splits": {}
}

for split_name, selected_groups in split_groups.items():
    split_data["splits"][split_name] = []

    for group in selected_groups:
        a, b = group
        group_id = f"{a}_{b}"

        videos = [
            (real_files[a], 0),
            (real_files[b], 0),
            (groups[group][0], 1),
            (groups[group][1], 1)
        ]

        split_data["splits"][split_name].append({
            "group_id": group_id,
            "videos": [file.name for file, label in videos]
        })

        for file, label in videos:
            rows.append({
                "video_name": file.name,
                "video_path": str(file),
                "label": label,
                "class": "fake" if label == 1 else "real",
                "group_id": group_id,
                "split": split_name
            })

df = pd.DataFrame(rows)

output = Path("datasets/metadata")
output.mkdir(parents=True, exist_ok=True)

split_output = Path("datasets/splits")
split_output.mkdir(parents=True, exist_ok=True)

df.to_csv(output / "ffpp_inventory.csv", index=False)

with open(split_output / "ffpp_split.json", "w") as f:
    json.dump(split_data, f, indent=4)

print("\nDATASET SPLIT COMPLETED")

for split_name in ["train", "validation", "test"]:
    subset = df[df["split"] == split_name]

    print(f"\n{split_name.upper()}")
    print("Videos:", len(subset))
    print("Real:", (subset["label"] == 0).sum())
    print("Fake:", (subset["label"] == 1).sum())
    print("Groups:", subset["group_id"].nunique())

all_groups = [
    set(df[df["split"] == name]["group_id"])
    for name in ["train", "validation", "test"]
]

assert all_groups[0].isdisjoint(all_groups[1])
assert all_groups[0].isdisjoint(all_groups[2])
assert all_groups[1].isdisjoint(all_groups[2])

print("\nIDENTITY GROUP LEAKAGE CHECK: PASSED")
print("Inventory saved:", output / "ffpp_inventory.csv")
print("Split saved:", split_output / "ffpp_split.json")
