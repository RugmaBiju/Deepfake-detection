import json
from pathlib import Path

base = Path(r"C:\FaceForensics++")

split_path = Path("datasets/splits/ffpp_split.json")

with open(split_path, "r") as f:
    data = json.load(f)

test_groups = data["splits"]["test"]

original_videos = set()
fake_videos = set()

for group in test_groups:
    for video in group["videos"]:
        if "_" in Path(video).stem:
            fake_videos.add(video)
        else:
            original_videos.add(video)

print("TEST GROUPS:", len(test_groups))
print("EXPECTED ORIGINAL VIDEOS:", len(original_videos))
print("EXPECTED FAKE VIDEOS:", len(fake_videos))

categories = ["Face2Face", "FaceSwap", "NeuralTextures"]

for category in categories:
    folder = base / "manipulated_sequences" / category / "c23" / "videos"

    missing_originals = [
        v for v in original_videos
        if not (base / "original_sequences" / "youtube" / "c23" / "videos" / v).exists()
    ]

    missing_fakes = [
        v for v in fake_videos
        if not (folder / v).exists()
    ]

    print(f"\n{category}")
    print("Original videos found:", len(original_videos) - len(missing_originals))
    print("Manipulated videos found:", len(fake_videos) - len(missing_fakes))
    print("Missing manipulated videos:", missing_fakes)