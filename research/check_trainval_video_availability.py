
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFPP = Path(r"C:\FaceForensics++")

with open(ROOT / "datasets" / "splits" / "ffpp_split.json") as f:
    data = json.load(f)

original_folder = (
    FFPP / "original_sequences" / "youtube" / "c23" / "videos"
)

categories = ["Face2Face", "FaceSwap", "NeuralTextures"]

for split_name in ["train", "validation"]:

    print("\n" + "=" * 60)
    print(split_name.upper())
    print("=" * 60)

    groups = data["splits"][split_name]

    original_videos = set()
    fake_videos = set()

    for group in groups:
        for video in group["videos"]:
            if "_" in Path(video).stem:
                fake_videos.add(video)
            else:
                original_videos.add(video)

    missing_original = [
        v for v in sorted(original_videos)
        if not (original_folder / v).exists()
    ]

    print("Original videos expected:", len(original_videos))
    print("Missing originals:", missing_original)

    for category in categories:

        folder = (
            FFPP / "manipulated_sequences" /
            category / "c23" / "videos"
        )

        missing = [
            v for v in sorted(fake_videos)
            if not (folder / v).exists()
        ]

        print("\n", category)
        print("Fake videos expected:", len(fake_videos))
        print("Available:", len(fake_videos) - len(missing))
        print("Missing count:", len(missing))
        print("Missing files:", missing)
