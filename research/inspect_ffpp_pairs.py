
from pathlib import Path
from collections import defaultdict

real_path = Path(r"C:\FaceForensics++\original_sequences\youtube\c23\videos")
fake_path = Path(r"C:\FaceForensics++\manipulated_sequences\Deepfakes\c23\videos")

real_ids = {p.stem for p in real_path.glob("*.mp4")}
fake_files = list(fake_path.glob("*.mp4"))

parent = {identity: identity for identity in real_ids}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a

valid_pairs = []
invalid_pairs = []

for file in fake_files:
    parts = file.stem.split("_")

    if len(parts) != 2:
        invalid_pairs.append(file.name)
        continue

    a, b = parts

    if a not in real_ids or b not in real_ids:
        invalid_pairs.append(file.name)
        continue

    union(a, b)
    valid_pairs.append((a, b))

groups = defaultdict(list)

for identity in real_ids:
    groups[find(identity)].append(identity)

components = sorted(
    groups.values(),
    key=len,
    reverse=True
)

unordered_pairs = {tuple(sorted(pair)) for pair in valid_pairs}

print("\nDATASET SUMMARY")
print("Real identities:", len(real_ids))
print("Fake videos:", len(fake_files))
print("Valid fake pairs:", len(valid_pairs))
print("Unique unordered pairs:", len(unordered_pairs))
print("Invalid pairs:", invalid_pairs)

print("\nIDENTITY GROUP ANALYSIS")
print("Connected components:", len(components))

for i, group in enumerate(components, 1):
    print(f"Group {i}: {len(group)} identities -> {sorted(group)}")

print("\nLargest group size:", len(components[0]) if components else 0)
